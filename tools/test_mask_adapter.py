import sys
import os
from pathlib import Path

# add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch

from cat_seg.maskadapter.mask_adapter import MASKAdapterHead


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model configuration - choose a "large" clip model name so clip_dim=768
    cfg = dict(
        clip_model_name='clip_base',
        mask_in_chans=64,
        num_channels=128,
        use_checkpoint=False,
        num_output_maps=1,
    )

    model = MASKAdapterHead(**cfg).to(device)
    model.eval()

    # Create dummy inputs
    B = 2               # batch size
    N = 16               # number of masks per image
    clip_dim = 512      # matches 'clip_large' branch in MASKAdapterHead
    H = 64
    W = 64

    # clip_feature: [B, C, H, W]
    clip_feature = torch.randn(B, clip_dim, H, W, device=device)

    # masks: [B, N, Hmask, Wmask] - use same H/W for simplicity
    masks = (torch.rand(B, N, H, W, device=device) > 0.5).float()

    print('clip_feature.shape =', clip_feature.shape)
    print('masks.shape =', masks.shape)

    # Forward
    with torch.no_grad():
        outputs = model(clip_feature, masks)

    print('outputs.shape =', outputs.shape)
    # outputs shape should be [B, N * num_output_maps, H, W]
    expected_c = N * cfg['num_output_maps']
    assert outputs.shape[0] == B
    assert outputs.shape[1] == expected_c
    assert outputs.shape[2] == H and outputs.shape[3] == W

    print('MASKAdapterHead test passed.')


if __name__ == '__main__':
    main()
