#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2C-Net 环境 + 数据 + 模型 端到端自检
=====================================

在正式训练前运行本脚本，可以一次性定位绝大多数“装完却跑不起来”的问题。
它按顺序检查六项，任何一项失败都会明确告诉你原因和修法。

    cd <project_root>
    python deploy/verify_env.py --data-dir ./TMS --ckpt /home/wlb/pretrained/sam2/sam2_hiera_large.pt

    # 没有权重时也可以先跑（模型随机初始化，只验证形状与数据管线）
    python deploy/verify_env.py --data-dir ./TMS --skip-model
"""

import argparse
import os
import sys
import traceback

OK = "[ OK ]"
NO = "[FAIL]"
WN = "[WARN]"


def head(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root_default = os.path.dirname(here)
    ap.add_argument("--root", default=root_default, help="project root")
    ap.add_argument("--code-dir", default="", help="dir containing train.py (default <root>/S2C-Net/S2C-Net)")
    ap.add_argument("--data-dir", default="", help="dir containing train_data/ and val_data/")
    ap.add_argument("--ckpt", default="", help="path to sam2_hiera_large.pt")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-model", action="store_true", help="skip the forward-pass test")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    code_dir = os.path.abspath(args.code_dir) if args.code_dir else os.path.join(root, "S2C-Net", "S2C-Net")
    failures = []

    # ---------------------------------------------------------------- 1
    head("1. Python / PyTorch / CUDA")
    print("python      :", sys.version.replace("\n", " "))
    try:
        import torch
        print("torch       :", torch.__version__)
        print("torch.cuda  :", torch.version.cuda)
        try:
            import torchvision
            print("torchvision :", torchvision.__version__)
        except ImportError:
            failures.append("torchvision 未安装")
            print("torchvision : 未安装")
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() == False")
            print(NO, "CUDA 不可用。检查驱动、CUDA_VISIBLE_DEVICES、以及是否装成了 CPU 版 wheel")
        else:
            print(OK, "CUDA 可用，设备数 =", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                print("      GPU %d: %s  sm_%d%d  %.1f GB" % (
                    i, prop.name, prop.major, prop.minor, prop.total_memory / 2**30))
            if torch.cuda.device_count() != 1:
                print(WN, "train_helper.py 中 assert self.device_count == 1；多卡请用 --device 只暴露一张")
    except ImportError:
        traceback.print_exc()
        failures.append("torch 未安装")
        print("后续检查中断")
        return 1

    # ---------------------------------------------------------------- 2
    head("2. sam2 包与 Hydra 配置解析（对应补丁 1）")
    try:
        import sam2
        print(OK, "sam2:", sam2.__file__)
    except Exception as e:
        print(NO, "import sam2 失败:", e)
        print("     -> 忘记安装？在 sam2 仓库根目录执行：SAM2_BUILD_CUDA=0 pip install -e .")
        failures.append("import sam2 失败")
        return 1

    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    resolved = None
    for name in ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_l.yaml"):
        try:
            GlobalHydra.instance().clear()
            with initialize_config_module("sam2", version_base="1.2"):
                cfg = compose(config_name=name)
            has_model = "model" in cfg
            print("      compose(%-34s) -> model 节点: %s" % (name, "存在" if has_model else "缺失"))
            if has_model:
                print("      cfg.model._target_ =", cfg.model._target_)
                resolved = name
                break
        except Exception as e:
            print("      compose(%r) 异常: %s: %s" % (name, type(e).__name__, e))

    model_py = os.path.join(code_dir, "Model.py")
    if os.path.isfile(model_py):
        txt = open(model_py, encoding="utf-8").read()
        applied = 'model_cfg = "configs/sam2/sam2_hiera_l.yaml"' in txt
        print("      Model.py 配置路径补丁：%s" % ("已应用" if applied else "未应用"))
        if not applied:
            failures.append("Model.py 未打补丁（补丁 1）")
            print(NO, "请执行 python deploy/apply_patches.py --root", root)
    if resolved != "configs/sam2/sam2_hiera_l.yaml":
        print(WN, "只有 configs/sam2/sam2_hiera_l.yaml 能解析出 model 节点")
    else:
        print(OK, "Hydra 配置可正常解析")

    # ---------------------------------------------------------------- 3
    head("3. 数据集目录结构")
    if not args.data_dir:
        print(WN, "未提供 --data-dir，跳过")
    else:
        dd = os.path.abspath(args.data_dir)
        for split in ("train_data", "val_data"):
            di = os.path.join(dd, split, "images")
            dg = os.path.join(dd, split, "ground-truth")
            ni = len([f for f in os.listdir(di) if f.lower().endswith(".jpg")]) if os.path.isdir(di) else -1
            ng = len([f for f in os.listdir(dg) if f.lower().endswith(".mat")]) if os.path.isdir(dg) else -1
            flag = OK if (ni > 0 and ni == ng) else NO
            print("%s %-11s images=%-6d ground-truth=%-6d" % (flag, split, ni, ng))
            if ni <= 0:
                failures.append("%s 缺失或为空" % split)
            elif ni != ng:
                failures.append("%s 图像/标注意外不等" % split)
        print("      期望结构: <data-dir>/{train_data,val_data}/{images/*.jpg,ground-truth/*.mat}")

    # ---------------------------------------------------------------- 4
    head("4. 标注文件读取（对应补丁 2）")
    if args.data_dir:
        try:
            import scipy.io as sio
            dd = os.path.abspath(args.data_dir)
            dg = os.path.join(dd, "train_data", "ground-truth")
            files = sorted(f for f in os.listdir(dg) if f.lower().endswith(".mat"))
            if not files:
                failures.append("ground-truth 目录无 .mat")
                print(NO, "无 .mat 文件")
            else:
                stats = {"sub_points": 0, "sub_point": 0, "other": 0}
                for f in files[:200]:
                    m = sio.loadmat(os.path.join(dg, f))
                    for k in ("sub_points", "sub_point"):
                        if k in m:
                            stats[k] += 1
                            break
                    else:
                        stats["other"] += 1
                print(OK, "抽样 200 个 .mat 键名统计:", stats)
                sample = sio.loadmat(os.path.join(dg, files[0]))
                key = "sub_points" if "sub_points" in sample else "sub_point"
                kp = sample[key][:, [2, 3]]
                print("      %s -> key=%s shape=%s 前两点(x,y)=%s" % (
                    files[0], key, sample[key].shape, kp[:2].tolist()))
                if stats["other"]:
                    failures.append("存在既无 sub_points 也无 sub_point 的 .mat")
        except ImportError:
            print(NO, "scipy 未安装")
            failures.append("scipy 未安装")
        except Exception:
            traceback.print_exc()
            failures.append("读取 .mat 失败")
    else:
        print(WN, "未提供 --data-dir，跳过")

    # ---------------------------------------------------------------- 5
    head("5. Dataset / DataLoader 冒烟（真实样本过一遍）")
    if args.data_dir:
        os.chdir(code_dir)
        sys.path.insert(0, code_dir)
        try:
            from datasets.RSOC import Crowd_TMS
            ds = Crowd_TMS(os.path.join(os.path.abspath(args.data_dir), "train_data"),
                           args.crop_size, 8, "train")
            img, pts, gt = ds[0]
            print(OK, "train[0]  img=%s pts=%s gt=%s" % (tuple(img.shape), tuple(pts.shape), tuple(gt.shape)))
            assert img.shape[1:] == (args.crop_size, args.crop_size), "crop 尺寸不符"
            assert gt.shape[1:] == (args.crop_size // 8, args.crop_size // 8), "密度图下采样尺寸不符"
            ds_val = Crowd_TMS(os.path.join(os.path.abspath(args.data_dir), "val_data"),
                               args.crop_size, 8, "val")
            vimg, vcount, vname = ds_val[0]
            print(OK, "val[0]    img=%s count=%s name=%s" % (tuple(vimg.shape), vcount, vname))
        except Exception:
            traceback.print_exc()
            failures.append("Dataset 冒烟失败（补丁 2 是否已应用？）")
    else:
        print(WN, "未提供 --data-dir，跳过")

    # ---------------------------------------------------------------- 6
    head("6. 模型构建与单步前向")
    if args.skip_model:
        print(WN, "按 --skip-model 跳过")
    else:
        os.chdir(code_dir)
        sys.path.insert(0, code_dir)
        ckpt = os.path.abspath(args.ckpt) if args.ckpt else None
        if ckpt and not os.path.isfile(ckpt):
            print(NO, "权重文件不存在:", ckpt)
            failures.append("权重文件不存在")
            ckpt = None
        if ckpt is None:
            print(WN, "未提供有效 --ckpt，将随机初始化（仅验证结构，不代表论文结果）")
        try:
            import Model
            net = Model.S2C_Net(ckpt)
            n_all = sum(p.numel() for p in net.parameters())
            n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(OK, "参数总量 %.2fM，其中可训练 %.2fM" % (n_all / 1e6, n_tr / 1e6))
            net = net.cuda().eval()
            x = torch.randn(2, 3, args.crop_size, args.crop_size, device="cuda")
            with torch.no_grad():
                mu, mu_normed = net(x)
            print(OK, "前向输出 mu=%s  mu_normed=%s" % (tuple(mu.shape), tuple(mu_normed.shape)))
            print("      预测计数 =", float(mu.sum().item()))
            print(OK, "模型构建与前向通过")
        except ImportError as e:
            print(NO, "缺少依赖:", e)
            print("     -> pip install einops wandb")
            failures.append("缺少依赖: %s" % e)
        except Exception:
            traceback.print_exc()
            failures.append("模型构建/前向失败")

    # ---------------------------------------------------------------- 汇总
    head("自检汇总")
    if failures:
        print(NO, "共 %d 项问题：" % len(failures))
        for f in failures:
            print("      -", f)
        return 1
    print(OK, "全部检查通过，可以开始训练：")
    print("      cd %s" % code_dir)
    print("      python train.py --dataset TMS --data-dir %s --hiera_path %s --device %s" % (
        os.path.abspath(args.data_dir) if args.data_dir else "<TMS>",
        os.path.abspath(args.ckpt) if args.ckpt else "<sam2_hiera_large.pt>",
        args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
