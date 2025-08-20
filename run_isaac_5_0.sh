# docker pull nvcr.io/nvidia/isaac-sim:5.0.0

# GPU配置 - 可选值: "all", "device=0", "device=1", "device=0,1" 等
GPU_DEVICES="all"
# GPU_DEVICES="device=3"

# 获取当前用户名和目录
USER_NAME=$(whoami)
CURRENT_DIR=$(pwd)

# 检查代理设置（防呆机制）
if [[ -z "$HTTP_PROXY" && -z "$HTTPS_PROXY" ]]; then
    echo "错误: 未检测到代理设置！"
    echo "请先设置代理环境变量："
    exit 1
fi

echo "启动Isaac Sim容器..."
echo "用户: $USER_NAME"
echo "工作目录: $CURRENT_DIR"
echo "GPU设备: $GPU_DEVICES"

# 检查是否已存在同名容器
if docker ps -a --format "table {{.Names}}" | grep -q "^isaac-sim-${USER_NAME}$"; then
    echo "容器 isaac-sim-${USER_NAME} 已存在! 请先停止或删除它。"
    exit 1
fi

docker run --name isaac-sim-${USER_NAME} -d --runtime=nvidia --gpus "$GPU_DEVICES" -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -e "HTTP_PROXY=$HTTP_PROXY" \
    -e "HTTPS_PROXY=$HTTPS_PROXY" \
    -e "NO_PROXY=$NO_PROXY" \
    -e "http_proxy=$HTTP_PROXY" \
    -e "https_proxy=$HTTPS_PROXY" \
    -e "no_proxy=$NO_PROXY" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ${CURRENT_DIR}:/workspace/isaac_lab:rw \
    -w /workspace/isaac_lab \
    --entrypoint /bin/bash \
    nvcr.io/nvidia/isaac-sim:5.0.0 \
    -c "while true; do sleep 3600; done"

# 检查容器是否成功启动
if [ $? -eq 0 ]; then
    echo "✅ Isaac Sim容器已成功启动在后台！"
    echo "容器名称: isaac-sim-${USER_NAME}"
    echo ""
    echo "可以使用以下命令连接到容器："
    echo "  docker exec -it isaac-sim-${USER_NAME} bash"
    echo ""
    echo "或者使用VS Code的Dev Containers扩展连接到容器"
    echo ""
    echo "查看容器状态:"
    echo "  docker ps | grep isaac-sim-${USER_NAME}"
    echo ""
    echo "停止容器:"
    echo "  docker stop isaac-sim-${USER_NAME}"
else
    echo "❌ 容器启动失败！"
    exit 1
fi