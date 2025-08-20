from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Simple CartPole PPO Training")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval")
parser.add_argument("--video_length", type=int, default=200, help="Video recording interval")
parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
parser.add_argument("--save_interval", type=int, default=10000, help="Model save interval")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Always enable cameras for video recording
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


# ==============================================================================
# Environment Configuration and Implementation
# ==============================================================================

@configclass
class SimpleCartpoleEnvCfg(DirectRLEnvCfg):
    """Configuration for the simple cartpole environment with camera."""

    # Environment settings
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # Camera for video recording (single global camera)
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,  # Update every frame
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0)
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # Viewer (for non-headless mode)
    viewer = ViewerCfg(eye=(10.0, 0.0, 6.0), lookat=(0.0, 0.0, 2.0))

    # Reset conditions
    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]

    # Reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class SimpleCartpoleEnv(DirectRLEnv):
    """Simple CartPole environment with video recording capabilities."""

    cfg: SimpleCartpoleEnvCfg

    def __init__(self, cfg: SimpleCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices
        self._cart_dof_idx, _ = self._cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self._cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        # Joint state buffers
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

    def _setup_scene(self):
        """Setup the scene with cartpole, camera, and lighting."""
        # Create cartpole articulation
        self._cartpole = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Add assets to scene
        self.scene.articulations["cartpole"] = self._cartpole

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Create single global camera for video recording separately
        # This is done after scene setup to avoid indexing conflicts
        self._camera = Camera(self.cfg.camera_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before physics step."""
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the cartpole."""
        self._cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        """Get observations for the RL agent."""
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards."""
        return compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = self._cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint positions with random pole angle
        joint_pos = self._cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self._cartpole.data.default_joint_vel[env_ids]

        # Reset root state
        default_root_state = self._cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update buffers
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Write to simulation
        self._cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""
        # Reset the base environment (handles cartpole reset)
        obs_dict, extras = super().reset(seed=seed, options=options)

        camera_pos = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32, device=self.device)
        target_pos = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=self.device)
        self._camera.set_world_poses_from_view(camera_pos, target_pos)

        # Don't reset camera since it's global and not environment-specific

        return obs_dict, extras

    def get_camera_image(self) -> np.ndarray:
        """Get camera image for video recording."""
        if hasattr(self, '_camera') and self._camera is not None:
            rgb_data = self._camera.data.output["rgb"]
            if rgb_data is not None:
                # Single camera image - Isaac Lab camera output is already uint8
                image = rgb_data[0].cpu().numpy()
                assert image.dtype == np.uint8
                return image
        return None


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """Compute rewards for the cartpole task."""
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward


# ==============================================================================
# PPO Algorithm Implementation
# ==============================================================================

class PPOPolicy(nn.Module):
    """PPO policy network with separate actor and critic."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor):
        """Forward pass through the network."""
        features = self.feature_net(obs)

        # Actor output
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)

        # Critic output
        value = self.critic(features)

        return action_mean, action_std, value.squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Get action and value, optionally evaluate a given action."""
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value


class PPOTrainer:
    """PPO trainer implementation."""

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
        indices = torch.randperm(batch_size)

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

    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.recording = False
        self.frame_count = 0

        os.makedirs(save_dir, exist_ok=True)

    def start_recording(self, filename: str, resolution: tuple):
        """Start recording a video."""
        if self.writer is not None:
            self.stop_recording()

        video_path = os.path.join(self.save_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        self.writer.release()
        self.writer = None
        self.recording = False
        self.frame_count = 0
        print("[VideoRecorder] Stopped recording")


# ==============================================================================
# Training Loop
# ==============================================================================

def main():
    """Main training function."""
    print("=" * 80)
    print("Simple CartPole PPO Training")
    print("=" * 80)

    # Create environment
    env_cfg = SimpleCartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = SimpleCartpoleEnv(cfg=env_cfg)

    # Training parameters
    num_envs = env.num_envs
    num_steps = args_cli.num_steps
    total_timesteps = args_cli.max_steps
    obs_dim = env_cfg.observation_space
    action_dim = env_cfg.action_space

    # Create policy and trainer
    device = env.device
    policy = PPOPolicy(obs_dim, action_dim).to(device)
    trainer = PPOTrainer(policy)

    # Storage for rollouts
    obs_buffer = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buffer = torch.zeros((num_steps, num_envs, action_dim), device=device)
    log_probs_buffer = torch.zeros((num_steps, num_envs), device=device)
    rewards_buffer = torch.zeros((num_steps, num_envs), device=device)
    dones_buffer = torch.zeros((num_steps, num_envs), device=device)
    values_buffer = torch.zeros((num_steps, num_envs), device=device)

    # Setup video recording
    video_recorder = VideoRecorder(
        save_dir=f"./videos/cartpole_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Setup TensorBoard logging
    log_dir = f"./logs/simple_cartpole_ppo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters to TensorBoard
    hparams = {
        'num_envs': num_envs,
        'num_steps': num_steps,
        'total_timesteps': total_timesteps,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'learning_rate': trainer.optimizer.param_groups[0]['lr'],
        'clip_coef': trainer.clip_coef,
        'value_coef': trainer.value_coef,
        'entropy_coef': trainer.entropy_coef,
        'gamma': trainer.gamma,
        'gae_lambda': trainer.gae_lambda,
        'update_epochs': trainer.update_epochs,
        'minibatch_size': trainer.minibatch_size,
    }
    writer.add_hparams(hparams, {})

    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")

    # Training statistics
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    # Get initial observation
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print(f"Training started with {num_envs} environments")
    print(f"Target steps: {total_timesteps}")
    print(f"Device: {device}")
    print("-" * 80)

    # Calculate number of updates needed
    num_updates = total_timesteps // num_steps

    # Create main progress bar for training
    pbar_main = tqdm(total=total_timesteps, desc="Training Progress", unit="steps")

    update_count = 0
    while global_step < total_timesteps:
        # Collect rollouts with progress bar
        for step in tqdm(range(num_steps), desc=f"Rollout {update_count+1}/{num_updates}", leave=False):
            obs_buffer[step] = obs

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(obs)

            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value

            # Environment step
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            done = terminated | truncated

            rewards_buffer[step] = reward
            dones_buffer[step] = done.float()

            global_step += 1
            print(f'global step = {global_step}')

            if not video_recorder.recording:
                if global_step % args_cli.video_interval == 0:
                    image = env.get_camera_image()
                    video_recorder.start_recording(
                        f"training_step_{global_step:08d}.mp4",
                        (image.shape[1], image.shape[0])
                    )
                    video_recorder.add_frame(image)
            else:
                env._camera.update(env.cfg.sim.dt)
                image = env.get_camera_image()
                video_recorder.add_frame(image)
                if video_recorder.frame_count >= args_cli.video_length:
                    video_recorder.stop_recording()



            # Log episode info
            if "episode" in info:
                for idx, episode_info in enumerate(info["episode"]):
                    if episode_info is not None:
                        episode_rewards.append(episode_info["r"])
                        episode_lengths.append(episode_info["l"])

                        # Log individual episode metrics to TensorBoard
                        writer.add_scalar('episode/reward', episode_info["r"], len(episode_rewards))
                        writer.add_scalar('episode/length', episode_info["l"], len(episode_rewards))

        # Compute advantages and returns
        with torch.no_grad():
            next_value = policy.get_action_and_value(obs)[3]

        advantages_buffer, returns_buffer = trainer.compute_advantages(
            rewards_buffer, values_buffer, dones_buffer, next_value
        )

        # Update policy
        update_info = trainer.update(
            obs_buffer, actions_buffer, log_probs_buffer, returns_buffer, advantages_buffer
        )

        # Update progress bars
        pbar_main.update(num_steps)
        update_count += 1

        # Update main progress bar description with current metrics
        if episode_rewards:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            mean_reward = np.mean(recent_rewards)
            pbar_main.set_postfix({
                'Reward': f'{mean_reward:.2f}',
                'Policy Loss': f'{update_info["policy_loss"]:.4f}',
                'Value Loss': f'{update_info["value_loss"]:.4f}'
            })

        # TensorBoard logging (every update)
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

        # Logging
        if global_step % args_cli.log_interval == 0:
            elapsed_time = time.time() - start_time
            fps = global_step / elapsed_time

            # Calculate recent performance
            if episode_rewards:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            else:
                mean_reward = 0.0
                mean_length = 0.0

            # Log performance metrics to TensorBoard
            writer.add_scalar('performance/fps', fps, global_step)
            writer.add_scalar('performance/elapsed_time', elapsed_time, global_step)

            print(f"Step: {global_step:8d} | "
                  f"FPS: {fps:6.0f} | "
                  f"Reward: {mean_reward:8.2f} | "
                  f"Length: {mean_length:6.1f} | "
                  f"Policy Loss: {update_info['policy_loss']:.4f} | "
                  f"Value Loss: {update_info['value_loss']:.4f}")

        # Save model periodically
        if global_step % args_cli.save_interval == 0:
            model_path = f"./models/cartpole_ppo_step_{global_step:08d}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'global_step': global_step,
                'config': env_cfg,
            }, model_path)
            print(f"[Model] Saved checkpoint: {model_path}")

    # Final cleanup
    if video_recorder.recording:
        video_recorder.stop_recording()

    # Close progress bar
    pbar_main.close()

    # Close TensorBoard writer
    writer.close()

    # Save final model
    final_model_path = "./models/cartpole_ppo_final.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'global_step': global_step,
        'config': env_cfg,
    }, final_model_path)

    print("=" * 80)
    print("Training completed!")
    print(f"Final model saved: {final_model_path}")
    print(f"Total steps: {global_step}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]) if episode_rewards else 0:.2f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view training logs, run: tensorboard --logdir {log_dir}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()