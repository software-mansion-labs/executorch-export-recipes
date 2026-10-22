"""Building blocks of the MediaPipe face-landmark model.

Vendored from https://github.com/tiqq111/mediapipe_pytorch (facial_landmarks/
utils.py), reduced to the block the model needs: the cv2-backed image helpers
and the TFLite key map are not used by the export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FacialLMBasicBlock(nn.Module):
    """DepthwiseConv + Conv + PReLU, with downsampling and channel padding.

    TFLite pads the depthwise conv differently from PyTorch when the stride is
    2, hence the explicit `F.pad` on the residual and the separate max-pool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.depthwiseconv_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.prelu(self.depthwiseconv_conv(h) + x)


def reflect_pad_top_left(x):
    """`nn.ReflectionPad2d((1, 0, 1, 0))`, written as two slice-concats.

    Reflecting by one is just a copy of the second row/column, and the
    concat form is bit-identical. The ReflectionPad2d op decomposes to an
    arange/abs/sub/index gather that the XNNPACK partitioner leaves behind on
    the portable kernels; this form lowers into the delegate with the rest of
    the graph.
    """
    x = torch.cat([x[:, :, :, 1:2], x], dim=3)
    return torch.cat([x[:, :, 1:2, :], x], dim=2)
