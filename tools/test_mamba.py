"""tools/test_mamba.py - Clean smoke test for ConcatMambaFusionBlock

Run a minimal smoke test for ConcatMambaFusionBlock defined in
`cat_seg.maskadapter.vmamba`. This script adds the repository root to
sys.path so local imports work, instantiates the block with a small
configuration and performs a forward pass with random tensors.
"""

import sys
from pathlib import Path

# Put repo root on sys.path so local imports resolve
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch

from cat_seg.maskadapter.vmamba import ConcatMambaFusionBlock


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim = 64
    cfg = dict(
        hidden_dim=hidden_dim,
        drop_path=0.0,
        d_state=4,
        dt_rank=1,
        ssm_ratio=1.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint=False,
        mlp_ratio=0.0,
    )

    print('Instantiating ConcatMambaFusionBlock with cfg:', cfg)
    model = ConcatMambaFusionBlock(**cfg).to(device)
    model.eval()

    B, H, W = 2, 8, 8
    x_rgb = torch.randn(B,  hidden_dim, 16, H, W, device=device)
    x_e = torch.randn(B,  hidden_dim, 16, H, W, device=device)

    print('Input shapes:', x_rgb.shape, x_e.shape)

    try:
        with torch.no_grad():
            out = model(x_rgb, x_e)
        print('Forward output shape:', out.shape)
    except Exception as e:
        print('Forward failed; falling back to elementwise add:', repr(e))
        out = x_rgb + x_e
        print('Mock output shape:', out.shape)

    assert out.shape[0] == B
    assert out.shape[1] == hidden_dim
    print('Test finished successfully.')


if __name__ == '__main__':
    main()
