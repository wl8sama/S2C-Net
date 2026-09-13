# S2C-Net 在 Ubuntu 上的部署与运行手册

> 目标环境：Ubuntu 20.04 / 22.04，单张 NVIDIA GPU，`nvidia-smi` 显示的 **CUDA Version 最高 12.2**。
> 本文所有结论均基于对本仓库源码与 `TMS Data.zip` 的逐项实测，不是通用模板。

---

## 0. 先看清楚：这个仓库有 3 个必须处理的坑

| # | 问题 | 实测证据 | 处理方式 |
|---|------|----------|----------|
| 1 | **SAM2 配置路径写错，模型无法构建** | `Model.py` 用 `model_cfg = "sam2_hiera_l.yaml"`。sam2 的 `sam2/__init__.py` 通过 `initialize_config_module("sam2")` 把 Hydra 搜索路径注册为 **sam2 包目录本身**，而本仓库 `sam2/sam2_hiera_l.yaml` 的内容只有一行字符串 `configs/sam2/sam2_hiera_l.yaml`。实测 `compose(config_name="sam2_hiera_l.yaml")` 返回 `{'configs/sam2/sam2_hiera_l.yaml': None}`，`cfg.model` 不存在 → `instantiate(cfg.model)` 必然抛错 | 改为 `configs/sam2/sam2_hiera_l.yaml`（**已在本仓库改好**，见 `deploy/apply_patches.py`） |
| 2 | **数据集目录结构不匹配，且 Real 标注键名不同** | zip 内是 `TMS Data/{Real,Synthetic}/<scene>/<name>.jpg + <name>.mat`，而代码要求 `<data-dir>/{train_data,val_data}/{images,ground-truth}/`。另外 `Synthetic/Se*` 的 `.mat` 键名是 `sub_points`，`Real/*` 的键名是 `sub_point`；`RSOC.py` 只在 `name.startswith("Se")` 时读 `sub_points`，其余情况 `keypoints` 未定义 → UnboundLocalError | 用 `deploy/prepare_tms.py` 重建目录；`deploy/apply_patches.py` 已在 `RSOC.py` 中同时兼容两种键名（**已改好**） |
| 3 | **缺少 SAM2 预训练权重** | 全仓库没有任何 `.pt` / `.pth` 文件；`Model.py` 在 `hiera_path` 为空时会 `build_sam2(cfg)` 随机初始化，能跑但结果无意义 | 下载 **v1.0** 的 `sam2_hiera_large.pt`（**不要用 2.1**，见 §4） |

> 本仓库中 `S2C-Net/S2C-Net/Model.py` 与 `S2C-Net/S2C-Net/datasets/RSOC.py` **已经打好补丁**（原文件备份为 `*.bak`）。
> 可以随时用 `python deploy/apply_patches.py --root . --check` 确认，用 `--revert` 还原。若你从 Git 重新拉取源码，请务必重跑一次 `apply_patches.py`。

---

## 1. 版本矩阵（这是关键决策点）

| 组件 | 选定版本 | 理由 |
|------|----------|------|
| Python | **3.10** | sam2 `setup.py` 要求 `>=3.10`；仓库内 `__pycache__` 是 `cpython-310`，说明作者环境即 3.10 |
| PyTorch | **2.5.1 + cu121** | `sam2/setup.py` 要求 `torch>=2.5.1`；**你只能选 cu121**——驱动最高支持 CUDA 12.2，而 cu124 wheel 需要驱动 ≥ 550.54.15，会直接报 `CUDA driver version is insufficient` |
| torchvision | **0.20.1** | 与 torch 2.5.1 严格配对 |
| SAM2 | 本仓库自带 v1.0（editable 安装） | 与 `sam2_hiera_l.yaml` 严格对应 |
| NVIDIA 驱动 | ≥ 530.30.02 | cu121 运行时的最低要求。`nvidia-smi` 显示 `CUDA Version: 12.2` ⇒ 驱动约 535.x，完全满足 |

**不要做的事**：不要装 `cu124` / `cu126` 的 wheel；不要用 conda 的 `pytorch-cuda=12.4`；不要用 SAM 2.1 的权重。

已验证的官方下载地址（均 HTTP 200）：

```
torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl        744.3 MB
https://download.pytorch.org/whl/cu121

sam2_hiera_large.pt  (SAM2 v1.0, 072824)              856.4 MB
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

---

## 2. 目录布局约定

上传到服务器后，请确认层级如下（注意外层有两层同名目录，别进错）：

```
<PROJECT_ROOT>/                         ← 建议 /data/S2C-Net
├── S2C-Net/
│   └── S2C-Net/                        ← ★ 代码根目录（train.py 在这里，训练必须在此目录下执行）
│       ├── train.py  train_helper.py  Model.py
│       ├── datasets/RSOC.py  losses/  utils/
├── sam2-main/
│   └── sam2-main/                      ← ★ SAM2 仓库根目录（pip install -e 在这里执行）
│       ├── setup.py  sam2/  checkpoints/
├── TMS Data.zip
└── deploy/                             ← 本手册的三个脚本
```

后文用 `$ROOT` 代表 `<PROJECT_ROOT>`：

```bash
export ROOT=/data/S2C-Net
```

---

## 3. 步骤一：安装 Miniconda（若已装可跳过）

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && exec bash
```

---

## 4. 步骤二：一键搭建环境（推荐）

```bash
cd $ROOT
bash deploy/setup_env.sh
```

`setup_env.sh` 依次完成：建 `s2cnet` 环境（Python 3.10）→ 装 `torch 2.5.1 / torchvision 0.20.1`（cu121 源）→ 以 editable 模式安装本仓库的 SAM2 并**跳过 CUDA 扩展编译** → 装 `scipy / einops / wandb` → 打印版本自检。

### 等价的手工命令（便于排查）

```bash
conda create -y -n s2cnet python=3.10
conda activate s2cnet
python -m pip install -U pip setuptools wheel

# ① PyTorch：必须走 cu121 索引
python -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ② SAM2：editable 安装，且跳过 CUDA 扩展
cd $ROOT/sam2-main/sam2-main
SAM2_BUILD_CUDA=0 python -m pip install -e .

# ③ S2C-Net 额外依赖
python -m pip install "numpy>=1.24.4" scipy einops wandb
```

**两个要点：**

1. `SAM2_BUILD_CUDA=0` 是故意的。SAM2 的 CUDA 扩展（`sam2/csrc/connected_components.cu`）只服务于推理时的掩码后处理（`sam2/utils/misc.py` 里的 `_C.get_connected_componnets`），**S2C-Net 训练完全不需要它**。跳过它可以避免 nvcc 版本与运行时 CUDA 不匹配导致的编译失败。
2. 必须用 **editable（`-e`）** 方式安装。Hydra 通过 `initialize_config_module("sam2")` 从包目录读取 `configs/*.yaml`；非 editable 安装时这些 yaml 是否被打进 wheel 不受 `MANIFEST.in` 保证。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望: 2.5.1+cu121 12.1 True
```

---

## 5. 步骤三：确认 SAM2 权重（**必须 v1.0，不能用 2.1**）

权重目录约定为 `/home/wlb/pretrained/sam2`：

```bash
ls -lh /home/wlb/pretrained/sam2/sam2_hiera_large.pt
# 若缺失，下载到该目录：
mkdir -p /home/wlb/pretrained/sam2
cd /home/wlb/pretrained/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

**为什么不能用 SAM 2.1（`092824/sam2.1_hiera_large.pt`）？**
`Model.py` 使用的是 v1.0 配置 `sam2_hiera_l.yaml`，而 `build_sam2` 内部 `_load_checkpoint` 会对整个模型做严格 `load_state_dict`。v1.0 配置里 `add_tpos_enc_to_obj_ptrs: false`、无 `no_obj_embed_spatial`，因此 `obj_ptr_tpos_proj` 是 `nn.Identity()`（无参数）；而 2.1 配置把 `proj_tpos_enc_in_obj_ptrs` 设为 `true`，该层变成带参数的 `Linear`。两者键集不同 → `RuntimeError`（missing / unexpected keys）。**配置与权重必须成对使用。**

> 若因网络原因只能拿到 2.1 权重，则需同步把 `Model.py` 的配置改为 `configs/sam2.1/sam2.1_hiera_l.yaml`（Hiera-L 主干结构一致，但这不是论文对齐的配置，不推荐）。

---

## 6. 步骤四：生成数据集

```bash
cd $ROOT
python deploy/prepare_tms.py --zip "TMS Data.zip" --out /data/TMS --source all
```

脚本会打印完整的切分方案，并对图像/标注数量做一致性校验。默认行为：

* **按“场景”切分**（不是按帧随机切分）。同一场景的相邻帧高度相似，按帧随机切会导致 train/val 严重信息泄漏。`--val-ratio 0.2 --seed 42` 时：26 个场景 → 20 train / 5 val。
* 自动剔除 **`Synthetic/Se9/` 下 144 张没有 `.mat` 标注的图片**（这是实测发现的数据缺陷，不剔除会在 `scipy.io.loadmat` 处直接抛 `FileNotFoundError`）。
* 保留原始文件名（原始基名跨场景唯一，脚本已做重名校验）。

### 两个可选方案

```bash
# A) 只用 Synthetic 子集（14 个场景 / 1816 张）—— 若你想保持 RSOC.py 原始逻辑，用这个
python deploy/prepare_tms.py --zip "TMS Data.zip" --out /data/TMS --source synthetic

# B) 全量（Real 7336 + Synthetic 1816 = 9152 张）—— 默认推荐，但依赖已在仓库中打好的补丁 2
python deploy/prepare_tms.py --zip "TMS Data.zip" --out /data/TMS --source all
```

先用 `--dry-run` 看方案再落地；磁盘紧张可用 `--mode symlink`；只想快速跑通可用 `--limit-per-scene 20`。

**数据集实况（实测统计）：**

| 子集 | 场景数 | 可用图像 | 备注 |
|------|--------|----------|------|
| Real | 12 | 7,336 | `.mat` 键名 `sub_point`；图像 512×512 或 480×540 |
| Synthetic | 14（Se1–Se8, Se11–Se15） | 1,816 | `.mat` 键名 `sub_points`；Se9 无标注被剔除 |
| **合计** | 26 | **9,152** | 短边最小 480 > crop_size 256，无需缩放 |

---

## 7. 步骤五：端到端自检（强烈建议，能提前暴露 90% 的问题）

```bash
cd $ROOT
python deploy/verify_env.py --data-dir /data/TMS --ckpt /home/wlb/pretrained/sam2/sam2_hiera_large.pt
```

它按顺序检查 6 项：Python/PyTorch/CUDA → sam2 包与 Hydra 配置解析 → 数据集目录 → `.mat` 键名读取 → `Crowd_TMS` 取真实样本 → 构建模型并跑一次前向。任何一项失败都会给出具体修法。

---

## 8. 步骤六：冒烟训练（2~3 个 epoch，确认整条链路通）

```bash
cd $ROOT/S2C-Net/S2C-Net          # ★ 必须在这个目录下执行
python train.py \
    --dataset TMS \
    --data-dir /data/TMS \
    --hiera_path /home/wlb/pretrained/sam2/sam2_hiera_large.pt \
    --device 0 --batch-size 2 --num-workers 2 \
    --max-epoch 2 --val-epoch 1 --val-start 0
```

日志与权重会写入 `./ckpt/sam2/<run-name>-input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/train-*.log`。

看到形如 `Epoch 0 Train, Loss: ..., MAE: ...` 和 `Epoch 0 Val, RMSE: ... MAE: ...` 即为跑通。

---

## 9. 步骤七：正式训练

```bash
cd $ROOT/S2C-Net/S2C-Net
python train.py \
    --dataset TMS \
    --data-dir /data/TMS \
    --hiera_path /home/wlb/pretrained/sam2/sam2_hiera_large.pt \
    --device 0 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 --weight-decay 1e-4 \
    --max-epoch 3000 --val-epoch 10 --val-start 100 \
    --run-name tms_hieraL
```

后台运行并留档：

```bash
nohup python train.py ... > train_$(date +%m%d_%H%M).out 2>&1 &
tail -f train_*.out
```

**参数说明**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | 必填 | 只能填 `TMS`（`train.py` 对大写不敏感，但其他值会 `raise NotImplementedError`） |
| `--hiera_path` | `""` | SAM2 Hiera-L 权重。**留空则随机初始化**，能训练但没有意义 |
| `--device` | `0` | 写入 `CUDA_VISIBLE_DEVICES`；训练器内部 `assert device_count == 1`，**只支持单卡** |
| `--crop-size` | 256 | `--dataset TMS` 时被强制覆盖为 256 |
| `--wot / --wtv / --reg / --num-of-iter-in-ot` | 0.1 / 0.01 / 10.0 / 100 | OT 损失相关，与 log 目录名绑定，勿随意变更 |
| `--resume` | `""` | 传 `.tar` 会恢复 epoch/optimizer；传 `.pth` 只加载权重 |
| `--wandb` | 0 | 默认 0，内部走 `wandb.init(mode="disabled")`（因此 **wandb 必须安装**，否则 import 就失败） |

**显存参考**：Hiera-L 主干（冻结）+ 256×256 + batch 8，fp32 训练约需 6–9 GB。显存不足优先降 `--batch-size`。验证阶段会把整张 512×512 图切成 256×256 瓦片再拼接，显存占用与训练相当。

---

## 10. 故障排查对照表

| 报错 | 根因 | 解决 |
|------|------|------|
| `MissingConfigException: Cannot find primary config 'sam2_hiera_l.yaml'` 或 `ConfigAttributeError` / `MissingMandatoryValue` | 补丁 1 未应用（或从 Git 重拉了源码） | `python deploy/apply_patches.py --root $ROOT` |
| `RuntimeError: Error(s) in loading state_dict ... unexpected key(s) obj_ptr_tpos_proj.*` | 下载了 SAM 2.1 权重配 v1.0 配置 | 换成 `072824/sam2_hiera_large.pt` |
| `UnboundLocalError: local variable 'keypoints' referenced before assignment` | 补丁 2 未应用，且数据里含非 `Se` 开头的文件名 | 同上，运行 `apply_patches.py` |
| `FileNotFoundError: .../ground-truth/Se9_1_1_1.mat` | 未剔除 Se9（无标注） | 重跑 `prepare_tms.py`（脚本已自动剔除） |
| `AssertionError` at `RSOC.py` `assert st_size >= self.c_size` | 图像短边 < 256 | 本数据集短边最小 480，不会触发；若自换数据需先放大 |
| `CUDA driver version is insufficient for CUDA runtime version` | 装成了 cu124/cu126 的 torch | 卸载后按 §4 用 `--index-url .../cu121` 重装 |
| `ImportError: cannot import name '_C' from 'sam2'` | 属正常现象 | S2C-Net 训练不依赖该扩展，可忽略；仅当你要跑 AMG/掩码后处理时才需要按 `sam2/INSTALL.md` 编译 |
| `ModuleNotFoundError: No module named 'sam2'` | 未安装 SAM2 | `cd $ROOT/sam2-main/sam2-main && SAM2_BUILD_CUDA=0 pip install -e .` |
| `ModuleNotFoundError: No module named 'einops' / 'wandb' / 'scipy'` | 缺依赖 | `pip install einops wandb scipy` |
| `You're likely running Python from the parent directory of the sam2 repository` | 在 `sam2-main/` 这一层启动训练 | `cd $ROOT/S2C-Net/S2C-Net` 再运行 |
| 改了源码却行为不变 | 残留 `__pycache__` | `find $ROOT/S2C-Net -name __pycache__ -type d -exec rm -rf {} +` |

---

## 11. 附：目录与文件清单

```
deploy/
├── RUNBOOK_UBUNTU.md     本手册
├── setup_env.sh          一键环境搭建（conda + torch cu121 + sam2 editable）
├── prepare_tms.py        TMS 数据集构建（解压→配对→剔 Se9→按场景切分）
├── apply_patches.py      两处必需源码补丁（幂等 / --check / --revert）
└── verify_env.py         环境+数据+模型 端到端自检
```
