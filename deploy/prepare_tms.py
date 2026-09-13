#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2C-Net / TMS 数据集准备脚本
============================

把 `TMS Data.zip` 转换成 train.py 所要求的目录结构：

    <out>/train_data/images/*.jpg
    <out>/train_data/ground-truth/*.mat
    <out>/val_data/images/*.jpg
    <out>/val_data/ground-truth/*.mat

背景（基于对本仓库数据的实测）：
  * zip 内原始结构为 `TMS Data/{Real,Synthetic}/<scene>/<name>.jpg` + `<name>.mat`
    —— 图像与标注同目录、按场景划分，与代码期望的 images//ground-truth/ 不一致，必须转换。
  * `Synthetic/Se9/` 下的 144 张 jpg **没有对应 .mat 标注**，必须剔除，否则
    `scipy.io.loadmat` 会抛 FileNotFoundError。
  * `Synthetic/Se*/*.mat` 的键名是 `sub_points`，`Real/*/*.mat` 的键名是 `sub_point`。
    RSOC.py 只读 `sub_points`，因此使用 Real 子集前必须先执行 deploy/apply_patches.py。
  * 全部图像尺寸为 512x512 或 480x540，短边 >= 480 > crop_size(256)，满足
    RSOC.py 中 `assert st_size >= self.c_size` 的断言，无需额外缩放。

用法示例
--------
# 全量（Real + Synthetic），按场景 8:2 切分，复制文件
python deploy/prepare_tms.py --zip "TMS Data.zip" --out ./TMS --source all

# 仅用 Synthetic 子集（代码零改动即可跑通）
python deploy/prepare_tms.py --zip "TMS Data.zip" --out ./TMS --source synthetic

# 冒烟测试：每个场景只取 4 张，用软链接（不占额外磁盘）
python deploy/prepare_tms.py --zip "TMS Data.zip" --out ./TMS_smoke \
       --source all --limit-per-scene 4 --mode symlink
"""

import argparse
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict

ZIP_PREFIX = "TMS Data/"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build S2C-Net TMS train/val directory layout from TMS Data.zip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zip", default="TMS Data.zip", help="path to TMS Data.zip")
    p.add_argument("--out", default="./TMS", help="output root (will contain train_data/ and val_data/)")
    p.add_argument("--source", default="all", choices=["all", "real", "synthetic"],
                   help="which subset to use")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="fraction of SCENES held out for validation")
    p.add_argument("--seed", type=int, default=42, help="random seed for the scene-level split")
    p.add_argument("--mode", default="copy", choices=["copy", "symlink", "hardlink"],
                   help="how to materialise files")
    p.add_argument("--limit-per-scene", type=int, default=0,
                   help="0 = use all; >0 keeps only N images per scene (smoke test)")
    p.add_argument("--scenes", default="", help="comma separated scene whitelist, e.g. Se1,Se2")
    p.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    return p.parse_args()


def index_zip(zip_path):
    """扫描 zip，返回 scenes[(kind, scene)] = [(basename, jpg_member, mat_member), ...]"""
    if not os.path.isfile(zip_path):
        sys.exit("[FATAL] zip not found: %s" % zip_path)

    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()

    jpgs = {}
    mats = set()
    for m in members:
        if not m.startswith(ZIP_PREFIX):
            continue
        low = m.lower()
        if low.endswith(".jpg"):
            jpgs[m] = True
        elif low.endswith(".mat"):
            mats.add(m)

    scenes = defaultdict(list)
    missing_mat = defaultdict(int)
    for jpg in jpgs:
        rel = jpg[len(ZIP_PREFIX):]              # Real/Boston1/xxx.jpg
        parts = rel.split("/")
        if len(parts) != 3:
            continue
        kind, scene, fname = parts
        base = os.path.splitext(fname)[0]
        mat = "%s%s/%s/%s.mat" % (ZIP_PREFIX, kind, scene, base)
        if mat not in mats:
            missing_mat[(kind, scene)] += 1
            continue
        scenes[(kind, scene)].append((base, jpg, mat))

    for k in scenes:
        scenes[k].sort(key=lambda t: t[0])
    return scenes, missing_mat


def main():
    args = parse_args()
    random.seed(args.seed)

    scenes, missing_mat = index_zip(args.zip)

    if missing_mat:
        print("[warn] 以下场景存在无标注的图片，已自动剔除：")
        for (kind, scene), n in sorted(missing_mat.items()):
            print("        %-10s %-18s 无标注 jpg = %d" % (kind, scene, n))
        print()

    # 场景筛选
    if args.source == "real":
        allowed_kinds = {"Real"}
    elif args.source == "synthetic":
        allowed_kinds = {"Synthetic"}
    else:
        allowed_kinds = {"Real", "Synthetic"}

    whitelist = {s.strip() for s in args.scenes.split(",") if s.strip()}

    usable = {}
    for (kind, scene), items in scenes.items():
        if kind not in allowed_kinds:
            continue
        if whitelist and scene not in whitelist:
            continue
        if not items:
            continue
        usable[(kind, scene)] = items

    if not usable:
        sys.exit("[FATAL] 没有可用场景，请检查 --source / --scenes 参数")

    # 场景级切分（按场景切分可避免同一场景的相邻帧同时出现在 train/val，防止信息泄漏）
    keys = sorted(usable.keys())
    random.shuffle(keys)
    n_val = max(1, int(round(len(keys) * args.val_ratio)))
    if n_val >= len(keys):
        n_val = len(keys) - 1
    val_keys = set(keys[:n_val])
    train_keys = [k for k in keys if k not in val_keys]

    print("=" * 74)
    print("TMS 数据集切分方案  (source=%s, val_ratio=%.2f, seed=%d)" % (args.source, args.val_ratio, args.seed))
    print("=" * 74)
    for tag, kk in (("train", train_keys), ("val", sorted(val_keys))):
        n_img = sum(len(usable[k]) for k in kk)
        print("%-5s 场景数 %2d  图像数 %5d" % (tag, len(kk), n_img))
        for k in sorted(kk):
            print("        %-10s %-18s %5d" % (k[0], k[1], len(usable[k])))
    print()

    # 同名冲突检查
    seen = defaultdict(list)
    for k in usable:
        for base, _, _ in usable[k]:
            seen[base].append(k)
    dup = {b: v for b, v in seen.items() if len(v) > 1}
    if dup:
        sys.exit("[FATAL] 发现跨场景重名基名，需要重命名策略：%s" % list(dup.items())[:5])

    total = sum(len(v) for v in usable.values())
    print("合计可用图像 %d 张（Real %d / Synthetic %d）" % (
        total,
        sum(len(v) for k, v in usable.items() if k[0] == "Real"),
        sum(len(v) for k, v in usable.items() if k[0] == "Synthetic"),
    ))

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return

    tasks = []
    for k in usable:
        split = "val_data" if k in val_keys else "train_data"
        items = usable[k]
        if args.limit_per_scene > 0:
            items = items[: args.limit_per_scene]
        for base, jpg, mat in items:
            tasks.append((split, base, jpg, mat))

    dirs = set()
    for split in ("train_data", "val_data"):
        dirs.add(os.path.join(args.out, split, "images"))
        dirs.add(os.path.join(args.out, split, "ground-truth"))
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    def materialise(src_member, dst_path, zf):
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            return
        if args.mode == "symlink":
            os.symlink(os.path.abspath(os.path.join(args.zip_abs, src_member)), dst_path)
        else:
            with zf.open(src_member) as src, open(dst_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)

    args.zip_abs = os.path.dirname(os.path.abspath(args.zip))
    n_done = 0
    with zipfile.ZipFile(args.zip) as zf:
        for split, base, jpg, mat in tasks:
            d_img = os.path.join(args.out, split, "images", base + ".jpg")
            d_mat = os.path.join(args.out, split, "ground-truth", base + ".mat")
            try:
                materialise(jpg, d_img, zf)
                materialise(mat, d_mat, zf)
            except OSError as e:
                print("[warn] %s -> %s" % (e, d_img))
                continue
            n_done += 1
            if n_done % 500 == 0:
                print("  ... 已处理 %d/%d" % (n_done, len(tasks)))

    print("\n[done] 写入 %d 组图像/标注 -> %s" % (n_done, os.path.abspath(args.out)))
    for split in ("train_data", "val_data"):
        ni = len(os.listdir(os.path.join(args.out, split, "images")))
        ng = len(os.listdir(os.path.join(args.out, split, "ground-truth")))
        print("       %-11s images=%d  ground-truth=%d %s" % (
            split, ni, ng, "OK" if ni == ng else "<== 数量不匹配，请检查!"))
    print("\n下一步：python train.py --dataset TMS --data-dir %s ..." % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
