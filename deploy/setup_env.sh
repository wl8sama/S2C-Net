#!/usr/bin/env bash
# =============================================================================
# S2C-Net Ubuntu 环境搭建脚本
#
# 目标：Python 3.10 + PyTorch 2.5.1(cu121) + 本仓库自带 SAM2(editable) + 训练依赖
#
# 为什么锁死 cu121：
#   sam2/setup.py 要求 torch>=2.5.1；
#   而 GPU 驱动最高只支持 CUDA 12.2，cu124 的 wheel 需要驱动 >= 550.54.15，
#   会直接报 "CUDA driver version is insufficient"。CUDA 12.1 运行时可在
#   12.2 驱动上正常跑（小版本向后兼容），因此唯一正确选择是 torch 2.5.1 + cu121。
#
# 用法：
#   cd <PROJECT_ROOT>
#   bash deploy/setup_env.sh
#
# 可用环境变量覆盖：
#   ENV_NAME=s2cnet   ROOT=/data/S2C-Net   PY_VER=3.10
# =============================================================================
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-s2cnet}"
PY_VER="${PY_VER:-3.10}"
SAM2_DIR="$ROOT/sam2-main/sam2-main"
CODE_DIR="$ROOT/S2C-Net/S2C-Net"
TORCH_VER="2.5.1"
TV_VER="0.20.1"
CKPT_DIR="${CKPT_DIR:-/home/wlb/pretrained/sam2}"

echo "======================================================================"
echo " S2C-Net 环境搭建"
echo "----------------------------------------------------------------------"
echo " 项目根目录 : $ROOT"
echo " SAM2 仓库  : $SAM2_DIR"
echo " 代码目录   : $CODE_DIR"
echo " conda 环境 : $ENV_NAME (python $PY_VER)"
echo " torch      : $TORCH_VER+cu121 / torchvision $TV_VER"
echo " 权重目录   : $CKPT_DIR"
echo "======================================================================"

# ---- 前置检查 ---------------------------------------------------------------
[ -d "$SAM2_DIR/sam2" ]      || { echo "ERROR: 找不到 SAM2 包目录 $SAM2_DIR/sam2"; exit 1; }
[ -f "$CODE_DIR/train.py" ]  || { echo "ERROR: 找不到 $CODE_DIR/train.py"; exit 1; }
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: 未找到 conda。请先安装 Miniconda："
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh"
    echo "  bash miniconda.sh -b -p \$HOME/miniconda3 && eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 && {
    echo "---- 当前驱动状态 ----"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo "----------------------"
}

# ---- 1. 创建/激活环境 -------------------------------------------------------
eval "$(conda shell.bash hook)"
if conda env list | awk 'NF{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> 复用已存在的 conda 环境: $ENV_NAME"
else
    echo "==> 创建 conda 环境: $ENV_NAME (python $PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

# ---- 2. PyTorch（必须 cu121） ----------------------------------------------
echo "==> 安装 PyTorch $TORCH_VER + cu121"
python -m pip install "torch==$TORCH_VER" "torchvision==$TV_VER" \
    --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import sys, torch
print("   python :", sys.version.split()[0])
print("   torch  :", torch.__version__)
print("   cuda   :", torch.version.cuda)
if not torch.cuda.is_available():
    print("   [WARN] torch.cuda.is_available() == False")
    print("          请确认：驱动 >= 530.30.02、容器已挂载 GPU(--gpus all)、未设置错误的 CUDA_VISIBLE_DEVICES")
else:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("   GPU %d  : %s sm_%d%d %.1fGB" % (i, p.name, p.major, p.minor, p.total_memory / 2**30))
PY

# ---- 3. 安装本仓库的 SAM2（editable，跳过 CUDA 扩展） -----------------------
# SAM2 的 CUDA 扩展只用于推理期的掩码后处理，S2C-Net 训练不需要；
# 跳过它可以避免 nvcc 版本与运行时 CUDA 不匹配导致的编译失败。
echo "==> 安装 SAM2 (editable, SAM2_BUILD_CUDA=0)"
cd "$SAM2_DIR"
SAM2_BUILD_CUDA=0 python -m pip install -e .

# ---- 4. S2C-Net 额外依赖 ----------------------------------------------------
echo "==> 安装 S2C-Net 依赖 (scipy / einops / wandb)"
python -m pip install "numpy>=1.24.4" scipy einops wandb

# ---- 5. 关键依赖自检 --------------------------------------------------------
echo "==> 依赖自检"
python - <<'PY'
mods = ["torch", "torchvision", "numpy", "scipy", "einops", "wandb", "hydra", "omegaconf", "PIL", "sam2"]
bad = []
for m in mods:
    try:
        __import__(m)
        print("   [ OK ] %s" % m)
    except Exception as e:
        print("   [FAIL] %s -> %s" % (m, e))
        bad.append(m)
raise SystemExit(1 if bad else 0)
PY

# ---- 6. 收尾提示 ------------------------------------------------------------
cat <<EOF

======================================================================
环境搭建完成。

后续步骤：
  1) 确认 SAM2 v1.0 权重已就位（必须是 v1.0，不能用 2.1）：
       ls -lh $CKPT_DIR/sam2_hiera_large.pt
       # 若缺失，下载到该目录：
       mkdir -p $CKPT_DIR && cd $CKPT_DIR
       wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

  2) 生成数据集：
       cd $ROOT
       python deploy/prepare_tms.py --zip "TMS Data.zip" --out /data/TMS --source all

  3) 端到端自检：
       python deploy/verify_env.py --data-dir /data/TMS --ckpt $CKPT_DIR/sam2_hiera_large.pt

  4) 训练（必须在代码目录下执行）：
       cd $CODE_DIR
       python train.py --dataset TMS --data-dir /data/TMS \\
           --hiera_path $CKPT_DIR/sam2_hiera_large.pt --device 0 --batch-size 8

详细说明见 deploy/RUNBOOK_UBUNTU.md
======================================================================
EOF
