#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2C-Net 运行必需的两处源码补丁（幂等，可重复执行）
==================================================

补丁 1 —— `S2C-Net/S2C-Net/Model.py`
    `model_cfg = "sam2_hiera_l.yaml"`
  -> `model_cfg = "configs/sam2/sam2_hiera_l.yaml"`

  原因：sam2 的 `sam2/__init__.py` 通过
  `initialize_config_module("sam2", version_base="1.2")` 把配置搜索路径注册为
  **sam2 包目录本身**，因此 Hydra 的 config_name 必须相对于 `sam2/` 解析。
  而本仓库 `sam2/sam2_hiera_l.yaml` 并不是真正的配置，其内容只有一行字符串
  `configs/sam2/sam2_hiera_l.yaml`。实测 `compose(config_name="sam2_hiera_l.yaml")`
  返回的是 `{'configs/sam2/sam2_hiera_l.yaml': None}`，`cfg.model` 不存在，
  随后 `instantiate(cfg.model)` 必然报错（ConfigAttributeError / MissingConfigException）。
  改为 `configs/sam2/sam2_hiera_l.yaml` 后实测可正确解析出
  `cfg.model._target_ = sam2.modeling.sam2_base.SAM2Base`。

补丁 2 —— `S2C-Net/S2C-Net/datasets/RSOC.py`
    仅当 `name.startswith("Se")` 时才读取 `sub_points`，其余文件名下 `keypoints`
    未定义 -> UnboundLocalError。
  另外实测：`Synthetic/Se*/*.mat` 的键名是 `sub_points`(复数)，
  `Real/*/*.mat` 的键名是 `sub_point`(单数)。
  改后两种键名、两种文件命名都能正常加载（列索引 [2,3] 即 (x, y)，与原实现一致）。

用法
----
    python deploy/apply_patches.py --root .          # 应用补丁（自动备份 .bak）
    python deploy/apply_patches.py --root . --check  # 只检查当前状态
    python deploy/apply_patches.py --root . --revert # 从 .bak 还原
"""

import argparse
import os
import shutil
import sys

PATCHES = [
    dict(
        name="patch-1: SAM2 Hydra 配置路径",
        rel="S2C-Net/S2C-Net/Model.py",
        old='model_cfg = "sam2_hiera_l.yaml"',
        new='model_cfg = "configs/sam2/sam2_hiera_l.yaml"',
    ),
    dict(
        name="patch-2: TMS 标注键名 sub_points / sub_point + 取消 Se 前缀限制",
        rel="S2C-Net/S2C-Net/datasets/RSOC.py",
        old=(
            '        if name.startswith("Se"):\n'
            "            keypoints = sio.loadmat(gd_path)['sub_points'][:, [2, 3]]\n"
        ),
        new=(
            "        _mat = sio.loadmat(gd_path)\n"
            '        _key = "sub_points" if "sub_points" in _mat else "sub_point"\n'
            "        keypoints = _mat[_key][:, [2, 3]]\n"
        ),
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (contains S2C-Net/ and sam2-main/)")
    ap.add_argument("--check", action="store_true", help="report status only")
    ap.add_argument("--revert", action="store_true", help="restore from .bak")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rc = 0

    for p in PATCHES:
        path = os.path.join(root, p["rel"])
        print("=" * 72)
        print("%s\n  file: %s" % (p["name"], path))

        if not os.path.isfile(path):
            print("  [FATAL] 文件不存在，--root 是否正确？")
            rc = 2
            continue

        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        has_new = p["new"] in src
        has_old = p["old"] in src

        if args.check:
            if has_new:
                print("  [OK] 补丁已应用")
            elif has_old:
                print("  [todo] 尚未应用（可执行不带 --check 的命令）")
                rc = 1
            else:
                print("  [warn] 未找到匹配文本，源码可能已被手工修改，请人工确认")
                rc = 1
            continue

        if args.revert:
            bak = path + ".bak"
            if not os.path.isfile(bak):
                print("  [skip] 没有 .bak 备份")
                continue
            shutil.copyfile(bak, path)
            print("  [ok] 已从 %s 还原" % os.path.basename(bak))
            continue

        if has_new:
            print("  [skip] 已经是补丁后状态（幂等）")
            continue
        if not has_old:
            print("  [warn] 未找到目标文本，跳过；请参照 docstring 手工修改")
            rc = 1
            continue

        shutil.copyfile(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(p["old"], p["new"], 1))
        print("  [ok] 已应用（原文件备份为 %s.bak）" % os.path.basename(path))

    print("=" * 72)
    if not args.check and not args.revert:
        print("提示：修改后请删除残留的 __pycache__，避免旧字节码干扰：")
        print("  find %s/S2C-Net -name __pycache__ -type d -exec rm -rf {} +" % root)
    sys.exit(rc)


if __name__ == "__main__":
    main()
