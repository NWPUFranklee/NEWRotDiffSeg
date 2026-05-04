"""Compute GFLOPs and parameter count for a model built from a detectron2-style config.

Usage:
  python3 tools/compute_model_stats.py --config-file configs/vitb_384.yaml --weights output/model_0019999.pth

This script only builds the model and computes flops/params using fvcore.
It avoids running dataset/evaluation.
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from train_net import add_cat_seg_config, add_deeplab_config
from train_net import setup as train_setup
from train_net import Trainer


def build_and_load_model(cfg_file: str, weights: str):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-file", default=cfg_file)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(weights, resume=False)
    return cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default=None, help="cuda or cpu; default auto")
    args = ap.parse_args()

    cfg, model = build_and_load_model(args.config_file, args.weights)

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    model.to(device)
    model.eval()

    # choose input resolution from cfg if available
    try:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        if isinstance(min_size, (list, tuple)):
            H = W = int(min_size[0])
        else:
            H = W = int(min_size)
    except Exception:
        H = W = 384

    # wrapper to adapt to model(batch) signature used in training script
    class _FlopModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # x: [B,3,H,W]
            batch = []
            for i in range(x.shape[0]):
                batch.append({"image": x[i], "height": H, "width": W})
            return self.model(batch)

    wrapper = _FlopModelWrapper(model).to(device)
    dummy = torch.randn(1, 3, H, W, device=device)
    try:
        flops = FlopCountAnalysis(wrapper, dummy)
        total_flops = flops.total()
        print(f"Model GFLOPs (approx): {total_flops / 1e9:.4f} GFLOPs")
        try:
            print(flop_count_str(flops))
        except Exception:
            pass
    except Exception as e:
        print("Failed to compute Flops:", e)

    try:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {total_params} ({total_params/1e6:.3f} M)")
    except Exception as e:
        print("Failed to compute params:", e)


if __name__ == '__main__':
    main()
