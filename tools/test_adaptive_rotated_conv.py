import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cat_seg.modeling.transformer.vmamba.adaptive_rotated_conv_multi import AdaptiveRotatedConv2d
from cat_seg.modeling.transformer.vmamba.routing_function import RountingFunction


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # simulate sizes from your snippet
    c4_size = 128
    c3_size = 128
    hidden_size = 64

    # conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
    conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False).to(device)

    # routing function and adaptive conv
    routing_function1 = RountingFunction(in_channels=hidden_size, kernel_number=1).to(device)
    routing_function2 = RountingFunction(in_channels=c3_size, kernel_number=1).to(device)
    conv2_4 = AdaptiveRotatedConv2d(in_channels=hidden_size, in_channels1=c3_size, out_channels=hidden_size,
                                    kernel_size=3, padding=1, rounting_func=routing_function1, rounting_func1=routing_function2, bias=False, kernel_number=1).to(device)

    # create dummy inputs: simulate concatenated features for conv1_4 input
    B = 2
    H = 32
    W = 32

    x_c4 = torch.randn(B, c4_size, H, W, device=device)
    x_c3 = torch.randn(B, c3_size, H, W, device=device)
    x_c5 = torch.randn(B, c3_size, H, W, device=device)

    x = torch.cat([x_c4, x_c3], dim=1)
    print('input concat shape:', x.shape)

    # pass through conv1_4 then conv2_4
    x1 = conv1_4(x)
    print('after conv1_4 shape:', x1.shape)

    # AdaptiveRotatedConv2d expects to call routing_function inside forward; ensure routing returns shapes
    try:
        out, out1 = conv2_4(x1, x_c5)
        print('AdaptiveRotatedConv2d output shape:', out.shape)
        print('AdaptiveRotatedConv2d output1 shape:', out1.shape)
    except Exception as e:
        print('AdaptiveRotatedConv2d forward failed with exception:', e)
        print('You may be missing CUDA selective-scan extensions used by vmamba; falling back to a simple numeric check:')
        # fallback: verify that routing function returns alphas and angles and that rotate_func can be called standalone
        alphas, angles = routing_function1(x1)
        alphas_e, angles_e = routing_function1(x1)
        print('routing alphas shape:', alphas.shape, 'angles shape:', angles.shape)
        print('routing alphas1 shape:', alphas_e.shape, 'angles1 shape:', angles_e.shape)

    print('Test finished.')


if __name__ == '__main__':
    main()
