from __future__ import annotations

import argparse
import math
import os
import time
import torch
from collections.abc import Sequence
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Jetbot PPO Training")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=512, help="Number of steps per rollout")
parser.add_argument("--max_rollouts", type=int, default=None, help="Maximum number of rollouts (overrides max_steps)")
parser.add_argument("--video_interval", type=int, default=5, help="Video recording interval (every N rollouts)")
parser.add_argument("--save_interval", type=int, default=10, help="Model save interval (every N rollouts)")
parser.add_argument("--eval", action="store_true", help="Run evaluation mode instead of training")
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
parser.add_argument("--eval_rollouts", type=int, default=1, help="Number of rollouts for evaluation")
# Session control arguments for multi-environment training
parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
parser.add_argument("--session_id", type=int, default=0, help="Training session ID for logging")
parser.add_argument("--rollouts_offset", type=int, default=0, help="Offset for rollout counting (for resumed training)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate")
parser.add_argument("--trial_name", type=str, help="Trial name for consistent directory naming")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for all training artifacts")


# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Always enable cameras for video recording
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


NUM_ENV = args_cli.num_envs
FRAME_RATE = 120
DECIMATION = 12
assert FRAME_RATE % DECIMATION == 0
VIDEO_SPEEDUP = 3

FOREST_RADIUS = 7
TARGET_RADIUS = 1
SPAWN_MARGIN = 1
SPAWN_DISTANCE = 1
PILLAR_NUM = 100
PILLAR_WIDTH = 0.2
COLLISION_DISTANCE_THRES = 0.1

# 目标位置（2D坐标，x, y）
TARGET_POSITION_2D = [0.0, 0.0]

MAX_EPISODE_TIME = 50

# 离散速度档位：负数后退，正数前进，0停止
DISCRETE_SPEEDS = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]

# 目标到达判断阈值（距离目标点多少米内算到达）
GOAL_REACH_THRESHOLD = 0.5


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils import configclass


from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils


JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
        activate_contact_sensors=True,  # 启用碰撞传感器
    ),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = DECIMATION
    episode_length_s = MAX_EPISODE_TIME
    # - spaces definition
    action_space = 2  # 2 discrete actions: left wheel speed index, right wheel speed index
    # observation_space = 激光雷达数据 + 目标位置信息 + 其他状态
    # 激光雷达: 实际714个数据点 (从调试输出得出)
    # 目标位置: 距离(1) + 角度差(1) + 交叉项(1) + 前进速度(1) + 偏航角速度(1) = 5
    # 总计: 714 + 5 = 719
    observation_space = 719
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / FRAME_RATE, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=JETBOT_CONFIG.spawn,
        actuators=JETBOT_CONFIG.actuators,
        prim_path="/World/envs/env_.*/Robot"
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10, env_spacing=0.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # Camera for video recording (single global camera)
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,  # Update every frame
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0)
        ),
    )

    # 360度激光雷达传感器配置
    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot",
        mesh_prim_paths=["/World/combined_scene"],  # 使用组合mesh
        pattern_cfg=patterns.LidarPatternCfg(
            channels=6,
            vertical_fov_range=(-20.0, 10.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=3.0,
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.15),  # 在机器人头顶上方15cm
            rot=(1.0, 0.0, 0.0, 0.0),  # 无旋转
        ),
        ray_alignment="base",  # 跟随机器人姿态
        debug_vis=False,  # 默认关闭可视化，只在第0号环境开启
        max_distance=10.0,  # 最大检测距离10米
        update_period=0.1,  # 10Hz更新频率
    )

    # 随机柱子配置
    obstacles_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/obstacles_group/obstacle_{ENV_ID}_{OBJ_ID}",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 1.0),  # 细高的柱子：10cm x 10cm x 100cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 默认红色，稍后随机化
        ),
    )

    # 碰撞检测Lidar配置 - 使用密集的360度水平扫描来检测障碍物
    collision_lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot",
        mesh_prim_paths=["/World/combined_scene"],  # 使用组合mesh
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # 单通道，水平扫描
            vertical_fov_range=(0.0, 0.0),  # 水平扫描，pitch=0
            horizontal_fov_range=(0.0, 360.0),  # 360度扫描
            horizontal_res=5.0,
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),  # 在机器人底盘高度
            rot=(1.0, 0.0, 0.0, 0.0),  # 无旋转
        ),
        ray_alignment="base",  # 跟随机器人姿态
        debug_vis=False,  # 关闭可视化以提高性能
        max_distance=2.0,  # 最大检测距离2米，足够检测碰撞
        update_period=0.0,  # 每个物理步都更新
    )

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def __init__(self, cfg: MyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        
        # 使用全局定义的离散速度档位
        self.discrete_speeds = torch.tensor(DISCRETE_SPEEDS, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 创建障碍物组容器（所有机器人共享）
        import isaacsim.core.utils.prims as prim_utils
        prim_utils.create_prim("/World/obstacles_group", "Xform")

        # 创建随机分布的柱子（同时创建ray casting用的mesh）
        self._create_random_obstacles()

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # 创建目标位置的可视化光圈
        self._create_target_visualization()

        # 初始化激光雷达传感器
        self.lidar = RayCaster(self.cfg.lidar_cfg)
        self.scene.sensors["lidar"] = self.lidar

        # 初始化碰撞检测Lidar
        self.collision_lidar = RayCaster(self.cfg.collision_lidar_cfg)
        self.scene.sensors["collision_lidar"] = self.collision_lidar

        # 启用激光雷达可视化，但我们会在回调中过滤只显示第0号环境
        # 先关闭默认的debug可视化，我们使用自定义的
        self.lidar.set_debug_vis(False)
        self.collision_lidar.set_debug_vis(False)  # 碰撞lidar不需要可视化

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()

        # 目标点设置为全局定义的位置
        self.target_position = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.target_position[:, 0] = TARGET_POSITION_2D[0]  # x坐标
        self.target_position[:, 1] = TARGET_POSITION_2D[1]  # y坐标
        self.target_position[:, 2] = 0.0  # z坐标保持为0

        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5

        # 距离跟踪变量用于奖励计算
        self.prev_target_distances = torch.zeros((self.cfg.scene.num_envs,), device=self.device)
        self.initial_target_distances = torch.zeros((self.cfg.scene.num_envs,), device=self.device)
        
        # 目标到达状态跟踪（防止重复给奖励）
        self.goal_reached_this_step = torch.zeros((self.cfg.scene.num_envs,), dtype=torch.bool, device=self.device)

        self._camera = Camera(self.cfg.camera_cfg)

    def _create_random_obstacles(self):
        """在场景中创建随机分布的固定彩色柱子，同时创建ray casting用的mesh"""
        import random
        import trimesh
        import isaacsim.core.utils.prims as prim_utils
        import time

        # 使用当前时间作为随机种子，确保每次都不同
        current_seed = int(time.time() * 1000) % 2147483647  # 限制在int32范围内
        random.seed(current_seed)
        print(f"[INFO] 使用随机种子 {current_seed} 生成障碍物")

        num_obstacles = PILLAR_NUM
        combined_meshes = []

        # 先创建地面mesh用于ray casting
        eps = 1e-3
        ground_mesh = trimesh.creation.box(extents=[100.0, 100.0, 0.1])
        ground_mesh.apply_translation([0, 0, -0.05 - eps])

        combined_meshes.append(ground_mesh)
        for i in range(num_obstacles):
            # 在森林范围内随机位置，但避开目标位置附近的区域
            while True:
                x = random.uniform(-FOREST_RADIUS, FOREST_RADIUS)
                y = random.uniform(-FOREST_RADIUS, FOREST_RADIUS)
                
                # 计算到目标位置的距离
                target_x, target_y = TARGET_POSITION_2D[0], TARGET_POSITION_2D[1]
                distance_to_target = math.sqrt((x - target_x)**2 + (y - target_y)**2)
                
                # 确保不在目标位置附近的TARGET_RADIUS范围内
                if distance_to_target > TARGET_RADIUS:
                    break

            # 随机高度（0.5米到1.0米）
            height = random.uniform(0.5, 1.0)
            z = height / 2  # 柱子中心高度

            width = random.uniform(0.5, 1.5) * PILLAR_WIDTH
            depth = random.uniform(0.5, 1.5) * PILLAR_WIDTH

            # 随机颜色
            color = (random.random(), random.random(), random.random())

            # 创建场景中的固定障碍物（不参与物理模拟）
            obstacle_path = f"/World/obstacles_group/obstacle_{i}"
            prim_utils.create_prim(obstacle_path, "Xform", translation=(x, y, z))

            # 添加几何形状作为静态碰撞体（有碰撞属性但不会移动）
            obstacle_cfg = sim_utils.CuboidCfg(
                size=(width, depth, height),  # 使用随机的宽度、深度和高度
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                # 添加碰撞属性，使其能被contact sensor检测
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # 不添加rigid_props和mass_props，使其成为静态碰撞体
            )
            obstacle_cfg.func(obstacle_path + "/geometry", obstacle_cfg)

            # 同时创建对应的mesh用于ray casting
            pillar_mesh = trimesh.creation.box(extents=[width, depth, height])
            pillar_mesh.apply_translation([x, y, z])
            combined_meshes.append(pillar_mesh)

        self.combined_mesh = trimesh.util.concatenate(combined_meshes)

        # 合并所有mesh用于ray casting
        print(f"[INFO] 成功创建 {len(combined_meshes)} 个mesh（包括地面和{num_obstacles}个固定障碍物）")
        print(f"[INFO] 所有 {self.cfg.scene.num_envs} 个机器人共享同一个环境")

        # 直接创建ray casting用的mesh prim（不保存文件）
        self._create_raycasting_mesh()

    def _create_target_visualization(self):
        """创建目标位置的透明可视化光圈"""
        import isaacsim.core.utils.prims as prim_utils
        
        # 在全局定义的目标位置创建透明圆柱体作为目标指示器
        target_x, target_y = TARGET_POSITION_2D[0], TARGET_POSITION_2D[1]
        target_path = "/World/target_indicator"
        prim_utils.create_prim(target_path, "Xform", translation=(target_x, target_y, 0.05))
        
        # 创建圆柱体几何形状
        target_cfg = sim_utils.CylinderCfg(
            radius=GOAL_REACH_THRESHOLD,  # 使用目标检测半径
            height=2.0,
            # 透明绿色材质，半透明效果
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                opacity=0.2
            ),
            # 不添加碰撞属性，纯可视化
        )
        target_cfg.func(target_path + "/geometry", target_cfg)
        
        print(f"[INFO] 创建目标可视化光圈: 位置({target_x}, {target_y}, 0), 半径{GOAL_REACH_THRESHOLD}米")

    def _create_raycasting_mesh(self):
        """直接创建ray casting用的mesh prim（不保存临时文件）"""
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom
        from isaaclab.terrains.utils import create_prim_from_mesh

        prim_utils.create_prim("/World/combined_scene", "Xform")
        create_prim_from_mesh("/World/combined_scene/mesh", self.combined_mesh)
        mesh_prim = prim_utils.get_prim_at_path("/World/combined_scene/mesh")
        if mesh_prim.IsValid():
            imageable = UsdGeom.Imageable(mesh_prim)
            imageable.MakeInvisible()
        else:
            raise


    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""
        # Reset the base environment (handles cartpole reset)
        obs_dict, extras = super().reset(seed=seed, options=options)

        # 设置摄像机位置以观察从起始位置到目标位置的路径
        # 摄像机位置在场景的一侧，稍微抬高
        target_x, target_y = TARGET_POSITION_2D[0], TARGET_POSITION_2D[1]
        camera_pos = torch.tensor([[-FOREST_RADIUS-SPAWN_MARGIN-5.0, target_y, FOREST_RADIUS*0.8]], dtype=torch.float32, device=self.device)
        target_pos = torch.tensor([[(target_x-FOREST_RADIUS-SPAWN_MARGIN) / 2, target_y, 0.0]], dtype=torch.float32, device=self.device)
        self._camera.set_world_poses_from_view(camera_pos, target_pos)
        
        # 初始化距离跟踪变量
        self._init_distance_tracking()

        return obs_dict, extras
    
    def _init_distance_tracking(self):
        """初始化距离跟踪变量"""
        robot_positions = self.robot.data.root_pos_w
        target_distances = torch.norm(robot_positions[:, :2] - self.target_position[:, :2], dim=1)
        self.prev_target_distances = target_distances.clone()
        self.initial_target_distances = target_distances.clone()

    def _visualize_markers(self):
        # 只为第0号小车显示箭头
        if self.cfg.scene.num_envs > 0:
            # 获取第0号机器人的位置作为标记位置
            marker_location = self.robot.data.root_pos_w[0:1]  # 只取第0号环境

            # 第一个箭头：指向车体前方（与车体对齐）
            forward_marker_orientation = self.robot.data.root_quat_w[0:1]  # 只取第0号环境

            # 第二个箭头：指向目标点
            robot_position = self.robot.data.root_pos_w[0:1]  # 只取第0号环境
            target_direction = self.target_position[0:1] - robot_position
            target_direction[:, 2] = 0.0  # 忽略z轴差异
            target_distance = torch.norm(target_direction, dim=1, keepdim=True)
            target_distance_safe = torch.clamp(target_distance, min=1e-6)
            target_direction_norm = target_direction / target_distance_safe

            # 计算指向目标的角度
            target_yaw = torch.atan2(target_direction_norm[:, 1], target_direction_norm[:, 0]).reshape(-1, 1)
            command_marker_orientation = math_utils.quat_from_angle_axis(target_yaw, self.up_dir).squeeze().unsqueeze(0)

            # 组合位置和旋转数据
            loc = marker_location + self.marker_offset[0:1]  # 只使用第0号环境的offset
            loc = torch.vstack((loc, loc))  # 两个箭头使用相同位置
            rots = torch.vstack((forward_marker_orientation, command_marker_orientation))

            # 标记索引：0 = forward箭头，1 = command箭头
            indices = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

            self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()# + torch.ones_like(actions)
        self._visualize_markers()
        # 更新激光雷达数据
        self.lidar.update(dt=self.step_dt)
        # 自定义激光雷达可视化：只显示第0号环境
        self._custom_lidar_visualization()

        # 更新碰撞检测Lidar
        self.collision_lidar.update(dt=self.step_dt)
        collision_status = self.is_collision()
        if torch.any(collision_status):
            print(f"检测到碰撞! 环境: {torch.where(collision_status)[0].tolist()}")

    def _apply_action(self) -> None:
        # 动作是离散的索引，需要转换为实际的轮速
        # actions shape: (num_envs, 2) - 分别是左轮和右轮的速度档位索引
        left_wheel_speeds = self.discrete_speeds[self.actions[:, 0].long()]
        right_wheel_speeds = self.discrete_speeds[self.actions[:, 1].long()]
        
        # 组合成轮速向量
        wheel_velocities = torch.stack([left_wheel_speeds, right_wheel_speeds], dim=1)
        
        self.robot.set_joint_velocity_target(wheel_velocities, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        # 获取激光雷达数据
        lidar_data = self.get_lidar_data()

        # 计算到目标点的距离和方向信息
        robot_positions = self.robot.data.root_pos_w
        target_directions = self.target_position - robot_positions
        target_directions[:, 2] = 0.0  # 忽略z轴差异
        target_distances = torch.norm(target_directions, dim=1, keepdim=True)

        # 避免除零
        target_distances_safe = torch.clamp(target_distances, min=1e-6)
        target_directions_norm = target_directions / target_distances_safe

        # 计算车体前方向与目标方向的夹角
        target_alignment = torch.sum(self.forwards * target_directions_norm, dim=-1, keepdim=True)
        target_cross = torch.cross(self.forwards, target_directions_norm, dim=-1)[:,-1].reshape(-1,1)

        # 车体状态
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

        # 组合所有观测信息
        obs = torch.hstack((
            lidar_data,           # 激光雷达数据 (714维)
            target_distances,     # 到目标距离 (1维)
            target_alignment,     # 目标对齐度 (1维)
            target_cross,         # 目标交叉项 (1维)
            forward_speed,        # 前进速度 (1维)
            self.robot.data.root_ang_vel_b[:, 2:3]  # 偏航角速度 (1维)
        ))

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励：基于距离变化、碰撞惩罚和时间惩罚"""
        
        # 1. 距离变化奖励 - 主要奖励信号
        robot_positions = self.robot.data.root_pos_w
        current_target_distances = torch.norm(robot_positions[:, :2] - self.target_position[:, :2], dim=1)
        
        # 距离变化：负值表示距离减少（好），正值表示距离增加（坏）
        distance_change = current_target_distances - self.prev_target_distances
        
        # 距离变化奖励：距离减少时给正奖励，距离增加时给负奖励
        # 使用较大的权重，因为这是主要奖励信号
        distance_reward = -distance_change * 10.0  # 每米距离变化对应10点奖励/惩罚
        
        # 更新上一次的距离记录
        self.prev_target_distances = current_target_distances.clone()
        
        # 2. 到达目标奖励 - 只在刚到达的那一步给一次奖励
        arrival_bonus = torch.where(self.goal_reached_this_step, 
                                   100.0,  # 简单的固定奖励：到达目标给100分
                                   0.0)
        
        # 3. 碰撞惩罚
        is_collided = self.is_collision()
        collision_penalty = is_collided.float() * -100.0  # 碰撞时给予-100的重惩罚
        
        # 4. 时间惩罚 - 鼓励快速到达目标
        time_penalty = torch.full_like(distance_reward, -0.01)  # 每步-0.01的时间惩罚
        
        # 总奖励
        total_reward = (distance_reward + 
                       arrival_bonus + 
                       collision_penalty + 
                       time_penalty).reshape(-1, 1)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 时间超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 碰撞导致的终止
        collision_done = self.is_collision()

        # 到达目标点的终止条件
        robot_positions = self.robot.data.root_pos_w
        target_distances = torch.norm(robot_positions[:, :2] - self.target_position[:, :2], dim=1)
        goal_reached = target_distances < GOAL_REACH_THRESHOLD  # 使用全局阈值
        
        # 记录本步到达目标的环境（用于奖励函数同步）
        self.goal_reached_this_step = goal_reached.clone()

        # 任何一种情况都会导致环境终止
        terminated = collision_done | goal_reached
        truncated = time_out

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.cfg.scene.num_envs))
        super()._reset_idx(env_ids)

        # 让机器人在目标位置的反方向生成，确保有足够的训练距离
        default_root_state = self.robot.data.default_root_state[env_ids].clone()

        for i, env_id in enumerate(env_ids):
            # 在目标位置的左侧（负x方向）生成机器人，这样需要向右（正x方向）移动到目标
            # 增加训练距离和难度
            x = torch.rand(1).item() * SPAWN_MARGIN - SPAWN_MARGIN - SPAWN_DISTANCE - FOREST_RADIUS
            y = torch.rand(1).item() * FOREST_RADIUS * 2 - FOREST_RADIUS

            default_root_state[i, 0] = x
            default_root_state[i, 1] = y
            default_root_state[i, 2] = 0.05  # 稍微抬高避免卡在地面

        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        self._visualize_markers()
        
        # 重置距离跟踪变量（使用正确的新位置）
        robot_positions = self.robot.data.root_pos_w[env_ids]
        target_distances_1d = torch.norm(robot_positions[:, :2] - self.target_position[env_ids, :2], dim=1)
        self.prev_target_distances[env_ids] = target_distances_1d
        self.initial_target_distances[env_ids] = target_distances_1d
        
        # 重置目标到达状态
        self.goal_reached_this_step[env_ids] = False


    def _custom_lidar_visualization(self):
        """自定义激光雷达可视化：只显示第0号环境的激光线"""
        if not hasattr(self, 'lidar_viz_markers'):
            # 创建激光雷达可视化标记
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/LidarViz",
                markers={
                    "hit_points": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self.lidar_viz_markers = VisualizationMarkers(cfg=marker_cfg)

        if self.lidar.data.ray_hits_w is not None:
            # 只显示第0号环境的激光雷达可视化
            ray_hits_env0 = self.lidar.data.ray_hits_w[0]  # 只取第0号环境
            # 移除无穷大的点
            valid_mask = ~torch.any(torch.isinf(ray_hits_env0), dim=1)
            if torch.any(valid_mask):
                viz_points = ray_hits_env0[valid_mask]
                # 每隔5个点显示一个，减少可视化开销
                # viz_points = viz_points[::5]
                # 显示激光击中点
                orientations = torch.zeros((len(viz_points), 4), device=viz_points.device)
                orientations[:, 0] = 1.0  # w分量
                marker_indices = torch.zeros(len(viz_points), dtype=torch.int32, device=viz_points.device)
                self.lidar_viz_markers.visualize(viz_points, orientations, marker_indices=marker_indices)

    def get_lidar_data(self) -> torch.Tensor:
        """获取激光雷达距离数据"""
        if self.lidar.data.ray_hits_w is not None:
            # ray_hits_w 的形状应该是 (num_envs, num_rays, 3)
            ray_hits = self.lidar.data.ray_hits_w
            
            # 计算从原点到击中点的距离
            distances = torch.norm(ray_hits, dim=-1)  # shape: (num_envs, num_rays)
            
            # 将无穷大的值（未击中）设置为最大距离
            distances = torch.where(torch.isinf(distances),
                                   torch.tensor(self.cfg.lidar_cfg.max_distance, device=distances.device),
                                   distances)
            
            return distances
        else:
            # 如果激光雷达未初始化，返回默认值
            # 根据实际测量：714个点
            expected_lidar_points = 714
            return torch.zeros((self.cfg.scene.num_envs, expected_lidar_points), device=self.device)

    def is_collision(self) -> torch.Tensor:
        """使用碰撞检测Lidar检测机器人是否与障碍物过于接近

        Returns:
            torch.Tensor: 布尔张量，形状为(num_envs,)，True表示发生碰撞
        """
        if self.collision_lidar.data.ray_hits_w is not None:
            # 计算从机器人位置到击中点的距离
            robot_positions = self.robot.data.root_pos_w  # shape: (num_envs, 3)
            ray_hits = self.collision_lidar.data.ray_hits_w  # shape: (num_envs, num_rays, 3)

            # 计算距离
            distances = torch.norm(ray_hits - robot_positions.unsqueeze(1), dim=-1)  # shape: (num_envs, num_rays)

            # 将无穷大的值（未击中）设置为最大距离
            distances = torch.where(torch.isinf(distances),
                                   torch.tensor(self.cfg.collision_lidar_cfg.max_distance, device=distances.device),
                                   distances)

            min_distances = torch.min(distances, dim=-1)[0]  # shape: (num_envs,)
            is_collided = min_distances < COLLISION_DISTANCE_THRES

            return is_collided
        else:
            # 如果碰撞检测Lidar未初始化，返回False（无碰撞）
            return torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=self.device)

    def get_camera_image(self) -> np.ndarray | None:
        """Get camera image for video recording."""
        if self._camera is not None:
            rgb_data = self._camera.data.output["rgb"]
            if rgb_data is not None:
                # Single camera image - Isaac Lab camera output is already uint8
                image = rgb_data[0].cpu().numpy()
                assert image.dtype == np.uint8
                return image
        return None

    def update_camera(self):
        """Public method to update camera."""
        if self._camera is not None:
            self._camera.update(self.step_dt)


# ==============================================================================
# PPO Algorithm Implementation
# ==============================================================================

class PPOPolicy(nn.Module):
    """PPO policy network with CNN for lidar data and discrete actions for Jetbot navigation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, lidar_channels: int = 6, lidar_points: int = 119):
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.lidar_points = lidar_points
        self.num_discrete_speeds = len(DISCRETE_SPEEDS)  # 使用全局常量的长度
        
        # 激光雷达数据：6通道 * 119点 = 714个数据点（实际测量值）
        self.lidar_data_size = lidar_channels * lidar_points  # 6 * 119 = 714
        self.other_obs_size = 5  # 目标距离(1) + 目标对齐度(1) + 目标交叉项(1) + 前进速度(1) + 偏航角速度(1)
        
        # 验证观测空间维度
        expected_obs_dim = self.lidar_data_size + self.other_obs_size  # 714 + 5 = 719
        assert obs_dim == expected_obs_dim, f"Expected obs_dim={expected_obs_dim}, got {obs_dim}"
        
        # 验证动作空间维度（现在是2，分别对应左轮和右轮的离散动作）
        assert action_dim == 2, f"Expected action_dim=2 for discrete wheel actions, got {action_dim}"
        
        # Lidar CNN feature extractor - 将激光雷达数据处理为range image
        # 6个通道 x 120个点，我们可以重塑为 (6, 120) 或者 (6, 20, 6) 等
        # 这里我们使用 (6, 120) 的形状，即6个通道，每个通道120个距离值
        self.lidar_cnn = nn.Sequential(
            # 输入: (batch, 6, 120) - 6通道的range data
            nn.Conv1d(6, 32, kernel_size=5, padding=2),  # -> (batch, 32, 120)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # -> (batch, 64, 120)
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 64, 60)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # -> (batch, 128, 60)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(30),  # -> (batch, 128, 30)
            nn.Flatten(),  # -> (batch, 3840)
            nn.Linear(3840, 256),  # -> (batch, 256)
            nn.ReLU(),
        )
        
        # 其他观测数据的全连接处理
        self.other_obs_net = nn.Sequential(
            nn.Linear(self.other_obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # 合并特征的网络
        combined_features_size = 256 + 64  # lidar features + other features
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_features_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor heads (policy) - separate heads for left and right wheel discrete actions
        self.actor_left_wheel = nn.Linear(hidden_dim, self.num_discrete_speeds)  # 对应DISCRETE_SPEEDS中的档位
        self.actor_right_wheel = nn.Linear(hidden_dim, self.num_discrete_speeds)  # 对应DISCRETE_SPEEDS中的档位

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor):
        """Forward pass through the network."""
        batch_size = obs.shape[0]
        
        # 分离激光雷达数据和其他观测数据
        lidar_data = obs[:, :self.lidar_data_size]  # 前714个元素是激光雷达数据
        other_obs = obs[:, self.lidar_data_size:]   # 后5个元素是其他观测
        
        # 将激光雷达数据重塑为 (batch, channels, points) 格式
        # lidar_data: (batch, 714) -> (batch, 6, 119)
        lidar_range_data = lidar_data.view(batch_size, self.lidar_channels, self.lidar_points)
        
        # 通过1D CNN处理激光雷达数据
        lidar_features = self.lidar_cnn(lidar_range_data)
        
        # 处理其他观测数据
        other_features = self.other_obs_net(other_obs)
        
        # 合并特征
        combined_features = torch.cat([lidar_features, other_features], dim=1)
        features = self.feature_fusion(combined_features)

        # Actor outputs - logits for discrete actions
        left_wheel_logits = self.actor_left_wheel(features)
        right_wheel_logits = self.actor_right_wheel(features)

        # Critic output
        value = self.critic(features)

        return left_wheel_logits, right_wheel_logits, value.squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Get action and value, optionally evaluate a given action."""
        left_wheel_logits, right_wheel_logits, value = self.forward(obs)
        
        # Create categorical distributions for each wheel
        left_wheel_dist = Categorical(logits=left_wheel_logits)
        right_wheel_dist = Categorical(logits=right_wheel_logits)

        if action is None:
            # Sample actions
            left_action = left_wheel_dist.sample()
            right_action = right_wheel_dist.sample()
            action = torch.stack([left_action, right_action], dim=1)
        else:
            # Use provided actions
            left_action = action[:, 0].long()
            right_action = action[:, 1].long()

        # Calculate log probabilities
        left_log_prob = left_wheel_dist.log_prob(left_action)
        right_log_prob = right_wheel_dist.log_prob(right_action)
        log_prob = left_log_prob + right_log_prob

        # Calculate entropy
        left_entropy = left_wheel_dist.entropy()
        right_entropy = right_wheel_dist.entropy()
        entropy = left_entropy + right_entropy

        return action, log_prob, entropy, value


class PPOTrainer:
    """PPO trainer implementation for Jetbot."""

    def __init__(
        self,
        policy: PPOPolicy,
        learning_rate: float = 3e-4,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: int = 256,
        warmup_steps: int = 1000,
        total_timesteps: int | None = None,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        # Learning rate scheduler with warmup + cosine
        self.initial_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_timesteps = total_timesteps
        self.current_step = 0

    def update_learning_rate(self, global_step: int):
        """Update learning rate with warmup + cosine annealing."""
        self.current_step = global_step
        
        if self.total_timesteps is None:
            return
        
        if global_step < self.warmup_steps:
            # Warmup phase: linear increase from 0 to initial_lr
            lr = self.initial_lr * global_step / self.warmup_steps
        else:
            # Cosine annealing phase
            cosine_steps = self.total_timesteps - self.warmup_steps
            cosine_progress = (global_step - self.warmup_steps) / cosine_steps
            cosine_progress = min(1.0, cosine_progress)  # Clamp to [0, 1]
            
            # Cosine annealing from initial_lr to 0
            lr = 0.5 * self.initial_lr * (1 + math.cos(math.pi * cosine_progress))
        
        # Apply the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict: dict):
        """Load trainer state from checkpoint."""
        if 'optimizer_state_dict' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if 'current_step' in state_dict:
            self.current_step = state_dict['current_step']

    def state_dict(self) -> dict:
        """Get trainer state for checkpointing."""
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
        }

    def compute_advantages(self, rewards, values, dones, next_value):
        """Compute GAE advantages."""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, log_probs, returns, advantages):
        """Update the policy using PPO."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch dimensions
        batch_size = obs.size(0) * obs.size(1)
        obs = obs.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        log_probs = log_probs.view(batch_size)
        returns = returns.view(batch_size)
        advantages = advantages.view(batch_size)

        # Create indices for minibatch updates
        indices = torch.randperm(batch_size, device=obs.device)

        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        # Add progress bar for epochs and minibatches
        num_minibatches = batch_size // self.minibatch_size
        total_updates = self.update_epochs * num_minibatches

        with tqdm(total=total_updates, desc="PPO Update", leave=False) as pbar_update:
            for epoch in range(self.update_epochs):
                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_indices = indices[start:end]

                    # Get current policy outputs
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        obs[mb_indices], actions[mb_indices]
                    )

                    # Policy loss
                    ratio = torch.exp(new_log_probs - log_probs[mb_indices])
                    surr1 = ratio * advantages[mb_indices]
                    surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages[mb_indices]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = 0.5 * ((new_values - returns[mb_indices]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    policy_loss_total += policy_loss.item()
                    value_loss_total += value_loss.item()
                    entropy_loss_total += entropy_loss.item()

                    # Update progress bar
                    pbar_update.update(1)
                    pbar_update.set_postfix({
                        'Policy Loss': f'{policy_loss.item():.4f}',
                        'Value Loss': f'{value_loss.item():.4f}',
                        'Entropy Loss': f'{entropy_loss.item():.4f}'
                    })

        num_updates = self.update_epochs * (batch_size // self.minibatch_size)
        return {
            'policy_loss': policy_loss_total / num_updates,
            'value_loss': value_loss_total / num_updates,
            'entropy_loss': entropy_loss_total / num_updates,
        }


class VideoRecorder:
    """Simple video recorder for training visualization."""

    def __init__(self, save_dir: str, fps):
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.recording = False
        self.frame_count = 0

        os.makedirs(save_dir, exist_ok=True)

    def start_recording(self, filename: str, resolution: tuple):
        assert not self.recording, "Already recording a video."
        """Start recording a video."""
        if self.writer is not None:
            self.stop_recording()

        video_path = os.path.join(self.save_dir, filename)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Use cv2.VideoWriter.fourcc instead
        self.writer = cv2.VideoWriter(video_path, fourcc, self.fps, resolution)
        self.recording = True
        print(f"[VideoRecorder] Started recording: {video_path}")

    def add_frame(self, image: np.ndarray):
        """Add a frame to the current recording."""
        assert self.recording
        assert self.writer is not None
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.writer.write(image_bgr)
        self.frame_count += 1

    def stop_recording(self):
        """Stop the current recording."""
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        self.frame_count = 0
        print("[VideoRecorder] Stopped recording")

def evaluate_policy(policy_path: str, num_rollouts: int = 10, record_all: bool = True):
    """Evaluate a trained policy and generate videos."""
    print("=" * 80)
    print("Jetbot PPO Policy Evaluation")
    print("=" * 80)
    
    # Create environment
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = MyEnv(cfg=env_cfg)
    
    # Load trained policy
    device = env.device
    obs_dim = 719
    action_dim = 2
    policy = PPOPolicy(obs_dim, action_dim).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()  # Set to evaluation mode
    
    print(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    print(f"Rollout count: {checkpoint.get('rollout_count', 'unknown')}")
    
    # Setup video recording
    # 评估模式使用独立的时间戳
    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_dir = args_cli.output_dir if hasattr(args_cli, 'output_dir') and args_cli.output_dir else "./outputs"
    video_recorder = VideoRecorder(
        save_dir=f"{video_dir}/jetbot_eval_{eval_timestamp}/videos",
        fps=FRAME_RATE / DECIMATION * VIDEO_SPEEDUP
    )
    
    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    
    print(f"Starting evaluation with {num_rollouts} rollouts...")
    print(f"Recording videos: {'All rollouts' if record_all else 'First rollout only'}")
    print("-" * 80)
    
    for rollout in range(num_rollouts):
        print(f"\nRollout {rollout+1}/{num_rollouts}")
        
        # Reset environment
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        
        # Start video recording
        should_record = record_all or (rollout == 0)
        if should_record:
            env.update_camera()
            image = env.get_camera_image()
            if image is not None:
                video_recorder.start_recording(
                    f"eval_rollout_{rollout:03d}.mp4",
                    (image.shape[1], image.shape[0])
                )
                video_recorder.add_frame(image)
        
        rollout_rewards = []
        rollout_lengths = []
        step_count = 0
        
        # Run rollout with step-level progress bar
        with tqdm(total=args_cli.num_steps, desc=f"Steps", leave=False) as pbar:
            while step_count < args_cli.num_steps:
                # Get action from policy (deterministic evaluation)
                with torch.no_grad():
                    # Ensure obs is a tensor (policy observation)
                    assert isinstance(obs, torch.Tensor), "obs should be a tensor"
                    left_wheel_logits, right_wheel_logits, _ = policy.forward(obs)
                    # Use greedy action (argmax) for deterministic evaluation
                    left_action = torch.argmax(left_wheel_logits, dim=1)
                    right_action = torch.argmax(right_wheel_logits, dim=1)
                    action = torch.stack([left_action, right_action], dim=1)
                
                # Environment step
                obs_dict, reward, terminated, truncated, info = env.step(action)
                obs = obs_dict["policy"]
                done = terminated | truncated
                
                step_count += 1
                pbar.update(1)
                
                # Record video frame
                if should_record and video_recorder.recording:
                    env.update_camera()
                    image = env.get_camera_image()
                    if image is not None:
                        video_recorder.add_frame(image)
                
                # Collect episode statistics
                if "episode" in info:
                    for idx, episode_info in enumerate(info["episode"]):
                        if episode_info is not None:
                            episode_rewards.append(episode_info["r"])
                            episode_lengths.append(episode_info["l"])
                            rollout_rewards.append(episode_info["r"])
                            rollout_lengths.append(episode_info["l"])
                            
                            # Check success/failure
                            if episode_info["r"] > 90:  # Assume success if high reward (reached goal)
                                success_count += 1
                            elif episode_info["r"] < -90:  # Assume collision if very negative reward
                                collision_count += 1
                
                # Update progress bar with current episode count
                if rollout_rewards:
                    pbar.set_postfix({"Episodes": len(rollout_rewards)})
                
                # Break if all environments are done (optional early termination)
                if torch.all(done):
                    break
        
        # Stop video recording
        if should_record and video_recorder.recording:
            video_recorder.stop_recording()
        
        # Print rollout summary
        if rollout_rewards:
            avg_reward = np.mean(rollout_rewards)
            avg_length = np.mean(rollout_lengths)
            print(f"Rollout {rollout+1:2d}: Avg Reward = {avg_reward:6.2f}, Avg Length = {avg_length:5.1f}, Episodes = {len(rollout_rewards)}")
    
    # Final statistics
    print("=" * 80)
    print("Evaluation Results:")
    print(f"Total Rollouts: {num_rollouts}")
    print(f"Total Episodes: {len(episode_rewards)}")
    
    if episode_rewards:
        print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
        print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
        print(f"Success Rate: {success_count}/{len(episode_rewards)} ({100*success_count/len(episode_rewards):.1f}%)")
        print(f"Collision Rate: {collision_count}/{len(episode_rewards)} ({100*collision_count/len(episode_rewards):.1f}%)")
    
    print(f"Videos saved to: {video_recorder.save_dir}")
    print("=" * 80)
    
    env.close()

def eval():
    if not args_cli.checkpoint:
        print("Error: --checkpoint must be specified for evaluation mode")
        return
    evaluate_policy(
        policy_path=args_cli.checkpoint,
        num_rollouts=args_cli.eval_rollouts,
    )

def train():
    """Main training function for Jetbot PPO."""
    print("=" * 80)
    print("Jetbot PPO Navigation Training")
    if args_cli.session_id > 0:
        print(f"Session ID: {args_cli.session_id}")
    if args_cli.resume:
        print(f"Resuming from: {args_cli.resume}")
    print("=" * 80)

    # Training parameters
    num_envs = args_cli.num_envs
    num_steps = args_cli.num_steps
    
    total_timesteps = args_cli.max_rollouts * num_steps * num_envs
    target_rollouts = args_cli.max_rollouts
    print(f"Training target: {args_cli.max_rollouts} rollouts ({total_timesteps} total steps)")
    
    obs_dim = 719  # observation space dimension (714 lidar + 5 other)
    action_dim = 2  # action space dimension (left wheel index, right wheel index)

    # Create initial environment to get device info
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = MyEnv(cfg=env_cfg)
    device = env.device
    
    # Create policy and trainer
    policy = PPOPolicy(obs_dim, action_dim).to(device)
    trainer = PPOTrainer(
        policy, 
        learning_rate=args_cli.learning_rate,
        warmup_steps=args_cli.warmup_steps,
        total_timesteps=total_timesteps
    )

    # 初始化训练状态
    global_step = 0
    rollout_count = args_cli.rollouts_offset
    start_rollout = rollout_count
    
    # 如果有checkpoint，加载它
    if args_cli.resume and os.path.exists(args_cli.resume):
        print(f"Loading checkpoint from: {args_cli.resume}")
        checkpoint = torch.load(args_cli.resume, map_location=device, weights_only=False)
        
        # 加载模型状态
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # 加载训练器状态
        trainer.load_state_dict(checkpoint)
        
        # 恢复训练状态
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        if 'rollout_count' in checkpoint:
            rollout_count = max(rollout_count, checkpoint['rollout_count'])
            
        print(f"Resumed from global step: {global_step}")
        print(f"Resumed from rollout: {rollout_count}")
    else:
        print("Starting training from scratch")

    # Storage for rollouts
    obs_buffer = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buffer = torch.zeros((num_steps, num_envs, action_dim), device=device)
    log_probs_buffer = torch.zeros((num_steps, num_envs), device=device)
    rewards_buffer = torch.zeros((num_steps, num_envs), device=device)
    dones_buffer = torch.zeros((num_steps, num_envs), device=device)
    values_buffer = torch.zeros((num_steps, num_envs), device=device)

    # Setup video recording
    # 使用bash脚本提供的统一trial name，如果没有则生成一个
    trial_name = args_cli.trial_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    video_recorder = VideoRecorder(
        save_dir=f"{args_cli.output_dir}/{trial_name}/videos",
        fps=FRAME_RATE / DECIMATION * VIDEO_SPEEDUP
    )

    # Setup TensorBoard logging - 使用统一的trial name创建子目录
    trial_name = args_cli.trial_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args_cli.output_dir}/{trial_name}/logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters to TensorBoard
    hparams = {
        'num_envs': num_envs,
        'num_steps': num_steps,
        'total_timesteps': total_timesteps,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'learning_rate': args_cli.learning_rate,
        'warmup_steps': args_cli.warmup_steps,
        'clip_coef': trainer.clip_coef,
        'value_coef': trainer.value_coef,
        'entropy_coef': trainer.entropy_coef,
        'gamma': trainer.gamma,
        'gae_lambda': trainer.gae_lambda,
        'update_epochs': trainer.update_epochs,
        'minibatch_size': trainer.minibatch_size,
        'session_id': args_cli.session_id,
        'rollouts_offset': args_cli.rollouts_offset,
    }
    writer.add_hparams(hparams, {})

    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")

    # Training statistics - 手动跟踪episode统计信息
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    # 为每个环境跟踪当前episode的累积奖励和长度
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device)

    print(f"Training started with {num_envs} environments")
    print(f"Target rollouts: {target_rollouts} (starting from {start_rollout})")
    print(f"Device: {device}")
    print("-" * 80)

    # Create main progress bar for training
    remaining_rollouts = target_rollouts - (rollout_count - start_rollout)
    pbar_main = tqdm(total=remaining_rollouts, desc="Training Progress", unit="rollouts")

    while rollout_count < start_rollout + target_rollouts:
        # Reset environment at the start of each rollout
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]  # Extract policy observation tensor
        
        # 重置episode统计跟踪器
        current_episode_rewards.fill_(0)
        current_episode_lengths.fill_(0)
        
        # 决定是否开始录制新视频
        if (rollout_count - start_rollout) % args_cli.video_interval == 0 and not video_recorder.recording:
            env.update_camera()
            image = env.get_camera_image()
            if image is not None:
                video_recorder.start_recording(
                    f"training_rollout_{rollout_count:05d}.mp4",
                    (image.shape[1], image.shape[0])
                )
                video_recorder.add_frame(image)
        
        # Collect rollouts with progress bar
        for step in tqdm(range(num_steps), desc=f"Rollout {rollout_count}", leave=False):
            # Ensure obs is a tensor (policy observation)
            assert isinstance(obs, torch.Tensor), "obs should be a tensor"
            obs_buffer[step] = obs

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(obs)

            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value

            # Environment step
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]  # Extract policy observation tensor
            done = terminated | truncated

            rewards_buffer[step] = reward.flatten()
            dones_buffer[step] = done.float()

            global_step += 1

            # 如果正在录制视频，继续添加帧
            if video_recorder.recording:
                env.update_camera()
                image = env.get_camera_image()
                if image is not None:
                    video_recorder.add_frame(image)

            # 手动跟踪episode统计信息
            current_episode_rewards += reward.flatten()
            current_episode_lengths += 1
            
            # 当episode结束时记录统计信息
            for env_idx in range(num_envs):
                if done[env_idx]:  # 当某个环境的episode结束时
                    final_reward = current_episode_rewards[env_idx].item()
                    final_length = current_episode_lengths[env_idx].item()
                    
                    episode_rewards.append(final_reward)
                    episode_lengths.append(final_length)
                    
                    # Log individual episode metrics to TensorBoard
                    writer.add_scalar('episode/reward', final_reward, len(episode_rewards))
                    writer.add_scalar('episode/length', final_length, len(episode_lengths))
                    
                    # 重置该环境的统计
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
        
        # 处理视频录制：完成一个 rollout 后检查是否需要停止录制
        if video_recorder.recording:
            video_recorder.stop_recording()

        # Compute advantages and returns
        with torch.no_grad():
            # obs is already a tensor at this point (policy observation)
            assert isinstance(obs, torch.Tensor), "obs should be a tensor at this point"
            next_value = policy.get_action_and_value(obs)[3]

        advantages_buffer, returns_buffer = trainer.compute_advantages(
            rewards_buffer, values_buffer, dones_buffer, next_value
        )

        # Update policy
        update_info = trainer.update(
            obs_buffer, actions_buffer, log_probs_buffer, returns_buffer, advantages_buffer
        )

        # Update learning rate
        trainer.update_learning_rate(global_step)

        # Update progress bars
        pbar_main.update(1)
        rollout_count += 1

        # Update main progress bar description with current metrics
        if episode_rewards:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            mean_reward = np.mean(recent_rewards)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            pbar_main.set_postfix({
                'Reward': f'{mean_reward:.2f}',
                'LR': f'{current_lr:.2e}',
                'Policy Loss': f'{update_info["policy_loss"]:.4f}',
                'Value Loss': f'{update_info["value_loss"]:.4f}'
            })

        # TensorBoard logging (every rollout)
        writer.add_scalar('train/policy_loss', update_info['policy_loss'], global_step)
        writer.add_scalar('train/value_loss', update_info['value_loss'], global_step)
        writer.add_scalar('train/entropy_loss', update_info['entropy_loss'], global_step)

        if episode_rewards:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            mean_reward = np.mean(recent_rewards)
            writer.add_scalar('train/mean_episode_reward', mean_reward, global_step)
            writer.add_scalar('train/total_episodes', len(episode_rewards), global_step)

        if episode_lengths:
            mean_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            writer.add_scalar('train/mean_episode_length', mean_length, global_step)

        # Log learning rate and other training metrics
        current_lr = trainer.optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, global_step)
        writer.add_scalar('train/rollout_count', rollout_count, global_step)

        # Save model periodically (every N rollouts)
        if (rollout_count - start_rollout) % args_cli.save_interval == 0:
            model_path = f"{args_cli.output_dir}/{trial_name}/checkpoints/jetbot_ppo_rollout_{rollout_count:05d}_step_{global_step:08d}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型和训练器状态
            save_dict = {
                'policy_state_dict': policy.state_dict(),
                'global_step': global_step,
                'rollout_count': rollout_count,
                'config': env_cfg,
                'args': vars(args_cli),
            }
            save_dict.update(trainer.state_dict())
            
            torch.save(save_dict, model_path)
            print(f"[Model] Saved checkpoint: {model_path}")

    # Final cleanup
    if video_recorder.recording:
        video_recorder.stop_recording()

    # Close progress bar
    pbar_main.close()

    # Close TensorBoard writer
    writer.close()

    # Save final model
    final_model_path = f"{args_cli.output_dir}/{trial_name}/checkpoints/jetbot_ppo_session_{args_cli.session_id}_final.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    save_dict = {
        'policy_state_dict': policy.state_dict(),
        'global_step': global_step,
        'rollout_count': rollout_count,
        'config': env_cfg,
        'args': vars(args_cli),
    }
    save_dict.update(trainer.state_dict())
    
    torch.save(save_dict, final_model_path)

    print("=" * 80)
    print("Training session completed!")
    print(f"Session ID: {args_cli.session_id}")
    print(f"Final model saved: {final_model_path}")
    print(f"Total steps: {global_step}")
    print(f"Total rollouts: {rollout_count}")
    if episode_rewards:
        print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view training logs, run: tensorboard --logdir {log_dir}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    if args_cli.eval:
        eval()  # This will call evaluate_policy
    else:
        train()  # This will run training
    simulation_app.close()