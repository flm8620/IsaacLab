#!/bin/bash

# =============================================================================
# Jetbot PPO Training Controller Script
# 用于控制训练过程，定期重启环境以防止过拟合
# =============================================================================

# 设置环境变量
# 禁用DISPLAY以避免Isaac Sim尝试推流X11，这会干扰仿真运行
export DISPLAY=""

# 配置参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="/isaac-sim/python.sh"  # Isaac Sim Python执行器
PYTHON_SCRIPT="$SCRIPT_DIR/jetbot_ppo.py"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
LOG_DIR="$SCRIPT_DIR/logs/controller"
STATE_FILE="$SCRIPT_DIR/.training_state"

# 生成统一的trial name用于本次训练的所有子目录
TRIAL_NAME=$(date '+%Y%m%d_%H%M%S')
export TRIAL_NAME

# 训练参数
TOTAL_ROLLOUTS=200          # 总的rollout数量
ROLLOUTS_PER_SESSION=5      # 每个session的rollout数量（重启间隔）
NUM_ENVS=64                  # 环境数量
NUM_STEPS=512               # 每个rollout的步数
VIDEO_INTERVAL=5            # 视频录制间隔
SAVE_INTERVAL=5             # 模型保存间隔

# 创建必要的目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/controller.log"
}

# 读取训练状态
load_training_state() {
    if [[ -f "$STATE_FILE" ]]; then
        source "$STATE_FILE"
        log "加载训练状态: SESSION=$CURRENT_SESSION, TOTAL_ROLLOUTS_COMPLETED=$TOTAL_ROLLOUTS_COMPLETED"
    else
        CURRENT_SESSION=0
        TOTAL_ROLLOUTS_COMPLETED=0
        log "初始化新的训练状态"
    fi
}

# 保存训练状态
save_training_state() {
    cat > "$STATE_FILE" << EOF
CURRENT_SESSION=$1
TOTAL_ROLLOUTS_COMPLETED=$2
EOF
    log "保存训练状态: SESSION=$1, TOTAL_ROLLOUTS_COMPLETED=$2"
}

# 查找最新的checkpoint
find_latest_checkpoint() {
    local latest_checkpoint=""
    # 在outputs目录下查找所有trial的checkpoints，找到最新的
    if [[ -d "$OUTPUT_DIR" ]]; then
        latest_checkpoint=$(find "$OUTPUT_DIR" -name "jetbot_ppo_rollout_*.pth" -type f | sort -V | tail -1)
    fi
    echo "$latest_checkpoint"
}

# 启动训练session
start_training_session() {
    local session_id=$1
    local rollouts_completed=$2
    local checkpoint_path=$3
    
    log "启动训练会话 $session_id..."
    log "已完成rollouts: $rollouts_completed"
    log "使用checkpoint: ${checkpoint_path:-"none (从头开始)"}"
    
    # 构建命令行参数
    local cmd_args=(
        --num_envs "$NUM_ENVS"
        --num_steps "$NUM_STEPS"
        --max_rollouts "$ROLLOUTS_PER_SESSION"
        --video_interval "$VIDEO_INTERVAL"
        --save_interval "$SAVE_INTERVAL"
        --session_id "$session_id"
        --rollouts_offset "$rollouts_completed"
        --trial_name "$TRIAL_NAME"
        --output_dir "$OUTPUT_DIR"
        --headless
    )
    
    # 如果有checkpoint，添加resume参数
    if [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]]; then
        cmd_args+=(--resume "$checkpoint_path")
    fi
    
    # 启动训练
    log "执行命令: $PYTHON_EXECUTABLE $PYTHON_SCRIPT ${cmd_args[*]}"
    "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" "${cmd_args[@]}"
}

# 主训练循环
main() {
    log "开始Jetbot PPO训练控制器"
    log "Trial Name: $TRIAL_NAME"
    log "总rollouts: $TOTAL_ROLLOUTS"
    log "每会话rollouts: $ROLLOUTS_PER_SESSION"
    log "预计会话数: $((TOTAL_ROLLOUTS / ROLLOUTS_PER_SESSION))"
    
    # 加载训练状态
    load_training_state
    
    # 计算总会话数
    local total_sessions=$(( (TOTAL_ROLLOUTS + ROLLOUTS_PER_SESSION - 1) / ROLLOUTS_PER_SESSION ))
    
    # 从当前会话开始继续训练
    for (( session = CURRENT_SESSION; session < total_sessions; session++ )); do
        # 计算当前会话应该完成的rollouts数量
        local remaining_rollouts=$((TOTAL_ROLLOUTS - TOTAL_ROLLOUTS_COMPLETED))
        local session_rollouts=$((remaining_rollouts < ROLLOUTS_PER_SESSION ? remaining_rollouts : ROLLOUTS_PER_SESSION))
        
        if [[ $session_rollouts -le 0 ]]; then
            log "所有训练已完成"
            break
        fi
        
        log "开始会话 $((session + 1))/$total_sessions"
        log "本会话目标rollouts: $session_rollouts"
        
        # 查找最新的checkpoint
        local checkpoint_path=$(find_latest_checkpoint)
        
        # 启动训练会话
        start_training_session "$session" "$TOTAL_ROLLOUTS_COMPLETED" "$checkpoint_path"
        
        # 更新状态
        TOTAL_ROLLOUTS_COMPLETED=$((TOTAL_ROLLOUTS_COMPLETED + session_rollouts))
        CURRENT_SESSION=$((session + 1))
        save_training_state "$CURRENT_SESSION" "$TOTAL_ROLLOUTS_COMPLETED"
        
        log "会话 $((session + 1)) 完成，总完成rollouts: $TOTAL_ROLLOUTS_COMPLETED"
        
        # 检查是否已完成所有训练
        if [[ $TOTAL_ROLLOUTS_COMPLETED -ge $TOTAL_ROLLOUTS ]]; then
            log "所有训练已完成！"
            break
        fi
        
        # 会话间短暂休息
        log "会话间休息 3 秒..."
        sleep 3
    done
    
    log "训练控制器完成"
    
    # 清理状态文件
    rm -f "$STATE_FILE"
}

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    --total-rollouts N       总rollout数量 (默认: $TOTAL_ROLLOUTS)
    --rollouts-per-session N 每会话rollout数量 (默认: $ROLLOUTS_PER_SESSION)
    --num-envs N            环境数量 (默认: $NUM_ENVS)
    --num-steps N           每rollout步数 (默认: $NUM_STEPS)
    --reset                 重置训练状态，从头开始
    --status                显示当前训练状态
    --help                  显示此帮助信息

示例:
    $0                                    # 使用默认参数开始训练
    $0 --total-rollouts 500               # 设置总rollout数量
    $0 --rollouts-per-session 5           # 设置每会话rollout数量
    $0 --reset                           # 重置并从头开始训练
    $0 --status                          # 查看训练状态
EOF
}

# 显示状态
show_status() {
    if [[ -f "$STATE_FILE" ]]; then
        source "$STATE_FILE"
        log "当前训练状态:"
        log "  会话: $CURRENT_SESSION"
        log "  已完成rollouts: $TOTAL_ROLLOUTS_COMPLETED"
        log "  目标rollouts: $TOTAL_ROLLOUTS"
        log "  进度: $(( TOTAL_ROLLOUTS_COMPLETED * 100 / TOTAL_ROLLOUTS ))%"
    else
        log "未找到训练状态文件"
    fi
}

# 重置状态
reset_state() {
    log "重置训练状态..."
    rm -f "$STATE_FILE"
    log "训练状态已重置"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --total-rollouts)
            TOTAL_ROLLOUTS="$2"
            shift 2
            ;;
        --rollouts-per-session)
            ROLLOUTS_PER_SESSION="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --reset)
            reset_state
            exit 0
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 运行主程序
main
