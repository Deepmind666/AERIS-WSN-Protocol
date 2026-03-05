# Astra-Sim 环境与运行指令备忘

本文汇总目前在本机验证过的关键命令，方便重启或换环境后快速恢复。

## 1. WSL 环境

```bash
# 进入 WSL Ubuntu-24.04
wsl -d Ubuntu-24.04

# 切换到仓库根目录并激活虚拟环境
cd /mnt/c/astra-sim
source .venv/bin/activate

# 以 root 身份编译/运行（避免 Windows 分区的权限问题）
sudo -s
cd /mnt/c/astra-sim
source .venv/bin/activate
```

### 1.1 官方示例（Analytical 后端）

```bash
cd examples/run_scripts/analytical/congestion_unaware
bash Ring_reducescatter_4npus.sh
```

### 1.2 生成 MoE 轨迹

```bash
cd /mnt/c/astra-sim
python utils/generate_moe_trace.py \
  --num-experts 384 \
  --tokens-per-batch 98304 \
  --topk 2 \
  --hidden-dim 4096 \
  --expert-ffn-dim 16384 \
  --output-dir examples/workload/moe/huawei384_top2
```

### 1.3 运行华为 384 超节点示例

```bash
cd examples/run_scripts/analytical/congestion_unaware
bash MoE_Huawei384.sh
```

## 2. Docker 环境

### 2.1 启动 Docker 服务（WSL 内）

```bash
sudo service docker start
newgrp docker   # 可选：让当前 shell 获得 docker 组权限
```

### 2.2 构建镜像

```bash
cd /mnt/c/astra-sim
docker build -t astra-sim:latest -f Dockerfile .
```

### 2.3 启动容器

```bash
docker run -it --rm \
  --name astra-sim-env \
  --shm-size=8g \
  -v /mnt/c/astra-sim:/app/astra-sim \
  astra-sim:latest bash
```

进入容器后，仓库路径为 `/app/astra-sim`。

### 2.4 容器内运行 Quick Example

```bash
cd /app/astra-sim/examples/run_scripts/analytical/congestion_unaware
bash Ring_reducescatter_4npus.sh
```

如遇 CMakeCache 路径冲突，先执行：

```bash
cd /app/astra-sim/build/astra_analytical
rm -rf build   # 或执行 bash build.sh -l
```

然后重新运行脚本。

#### Protobuf 5.29 与 utf8_range 依赖

Docker 镜像内默认使用 Protobuf 5.29。若重建容器时遇到 `utf8_range` 或 GTest 相关错误，可在容器内执行：

```bash
apt update
apt install -y libgtest-dev
cd /tmp
git clone https://github.com/protocolbuffers/utf8_range.git
cd utf8_range
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/utf8_range \
  -DUTF8_RANGE_ENABLE_TESTS=OFF
cmake --build build --target install
export CMAKE_PREFIX_PATH=/opt/utf8_range:$CMAKE_PREFIX_PATH
export PROTOBUF_FROM_SOURCE=True
```

之后清理旧 build 并重新运行示例即可。
