"""MI-GAN inference generator, vendored from
https://github.com/Picsart-AI-Research/MI-GAN (lib/model_zoo/migan_inference.py).
MIT License, Copyright (c) 2024 Picsart AI Research (PAIR). Weights are MIT too
(LICENSE-WEIGHTS in the upstream repo).

Paper: Sargsyan et al., "MI-GAN: A Simple Baseline for Image Inpainting on
Mobile Devices", ICCV 2023.

The upstream file is reproduced between the VENDORED markers, verbatim except
for the one block marked `EXPORT PATCH` (the in-place ops upstream uses export
fine: torch.export functionalises them, verified bit-identical through the
ExecuTorch runtime, so they are left as-is). Our
additions (source pin, weight table, export wrapper) live below them.
"""
# --- VENDORED: begin (upstream lib/model_zoo/migan_inference.py) -------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class lrelu_agc:
    """
    The lrelu layer with alpha, gain and clamp
    """

    def __init__(self, alpha=0.2, gain=1, clamp=None):
        self.alpha = alpha
        if gain == 'sqrt_2':
            # EXPORT PATCH: upstream keeps np.sqrt(2) (a numpy float64). torch.export
            # traces numpy scalars as tensors, turning `if act_gain != 1` below into
            # data-dependent control flow. A Python float keeps it a constant.
            self.gain = float(np.sqrt(2))
        else:
            self.gain = gain
        self.clamp = clamp

    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


class Downsample2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            padding=1,
            bias=False,
            stride=2
        )
        f = setup_filter([1, 3, 3, 1], gain=1)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        x = self.filter(x)
        return x


class Upsample2d(nn.Module):
    def __init__(self, in_channels, resolution=None):
        super().__init__()
        self.nearest_up = nn.Upsample(scale_factor=2, mode='nearest')
        w = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float32)
        assert resolution is not None
        self.register_buffer('filter_const', w.repeat(1, 1, resolution//2, resolution//2))

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            bias=False
        )

        f = setup_filter([1, 3, 3, 1], gain=4)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        x = self.nearest_up(x)
        x = x * self.filter_const
        x = F.pad(x, pad=(2, 1, 2, 1))
        x = self.filter(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            bias=True,
            activation=None,
            resolution=None,
            use_noise=False,
            down=1,
            up=1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=in_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            groups=1
        )

        self.downsample = None
        if down > 1:
            self.downsample = Downsample2d(in_channels)

        self.upsample = None
        if up > 1:
            self.upsample = Upsample2d(out_channels, resolution=resolution)

        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.downsample is not None:
            x = self.downsample(x)
        x = self.conv2(x)
        if self.upsample is not None:
            x = self.upsample(x)

        if self.use_noise:
            noise = self.noise_const * self.noise_strength
            x = x.add_(noise)
        if self.activation is not None:
            x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        rgb_n=None,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256),
        down=2
    ):
        super().__init__()

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = nn.Conv2d(rgb_n, ic_n, 1)

        self.conv1 = SeparableConv2d(ic_n, ic_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(ic_n, oc_n, 3, activation=activation, down=down)
        self.activation = activation

    def forward(self, x, img):
        if self.fromrgb is not None:
            y = self.fromrgb(img)
            y = self.activation(y)
            x = x + y if x is not None else y

        feat = self.conv1(x)
        x = self.conv2(feat)
        return x, feat


class Encoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        ic_n=4,
        ch_base=32768,
        ch_max=512,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        self.encode_res = [2 ** i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n

        for idx, (resi, resj) in enumerate(
                zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)

            if idx == 0:
                block = EncoderBlock(hidden_ch_i, hidden_ch_j, rgb_n=ic_n, activation=activation)
            else:
                block = EncoderBlock(hidden_ch_i, hidden_ch_j, activation=activation)

            setattr(self, 'b{}'.format(resi), block)

        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = EncoderBlock(hidden_ch, hidden_ch, activation=activation, down=1)

    def forward(self, img):
        x = None
        feats = {}
        for resi in self.encode_res[0:-1]:
            block = getattr(self, 'b{}'.format(resi))
            x, feat = block(x, img)
            feats[resi] = feat

        x, feat = self.b4(x, img)
        feats[4] = feat

        return x, feats


class SynthesisBlockFirst(nn.Module):
    def __init__(
        self,
        oc_n,
        resolution,
        rgb_n=None,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256),
    ):
        """
        Args:
            oc_n: output channel number
        """
        super().__init__()
        self.resolution = resolution

        self.conv1 = SeparableConv2d(oc_n, oc_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=4, activation=activation)

        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)

    def forward(self, x, enc_feat):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        img = None
        if self.torgb is not None:
            img = self.torgb(x)

        return x, img


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        resolution,
        rgb_n,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        self.resolution = resolution

        self.conv1 = SeparableConv2d(ic_n, oc_n, 3, resolution=resolution, up=2, activation=activation, use_noise=True)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=resolution, up=1, activation=activation, use_noise=True)

        self.torgb = None
        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)
        self.upsample = Upsample2d(rgb_n, resolution=resolution)

    def forward(self, x, enc_feat, img):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        if img is not None:
            img = self.upsample(img)

        if self.torgb is not None:
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        return x, img


class Synthesis(nn.Module):
    def __init__(
        self,
        resolution=256,
        rgb_n=3,
        ch_base=32768,
        ch_max=512,
        activation=lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        block_res = [2 ** i for i in range(2, log2res + 1)]

        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = block_res

        hidden_ch = min(ch_base // block_res[0], ch_max)
        self.b4 = SynthesisBlockFirst(hidden_ch, resolution=4, rgb_n=rgb_n, activation=activation)

        for resi, resj in zip(block_res[:-1], block_res[1:]):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            block = SynthesisBlock(hidden_ch_i, hidden_ch_j, resolution=resj, rgb_n=rgb_n, activation=activation)
            setattr(self, 'b{}'.format(resj), block)

    def forward(self, x, enc_feats):
        x, img = self.b4(x, enc_feats[4])
        for res in self.block_res[1:]:
            block = getattr(self, f'b{res}')
            x, img = block(x, enc_feats[res], img)
        return img


class Generator(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()

        self.synthesis = Synthesis(resolution=resolution)
        self.encoder = Encoder(resolution=resolution)

    def forward(self, x):
        """
        Args:
            x: 4 channel rgb+mask
        """
        x, feats = self.encoder(x)
        img = self.synthesis(x, feats)
        return img
# --- VENDORED: end -----------------------------------------------------------

SOURCE_REPO = "Picsart-AI-Research/MI-GAN"
# Pinned to a commit, not `main`: the upstream repo is unversioned.
SOURCE_COMMIT = "2b793c5ece43f4253e32d4afc257120a5deed6f5"

# Upstream publishes the weights only in a public Google Drive folder
# (linked from its README). The file ids below are the stable per-file ids of
# that folder. The `drive.usercontent.google.com` host + `confirm=t` is what returns
# the bytes; the plain `drive.google.com/uc` link answers with an HTML
# "too large to virus-scan" interstitial for files over ~25 MB.
# Once the SWM HF repo (model.json `repo`) carries `weights/<checkpoint>`, prefer
# fetching from there and keep Drive as the fallback.
WEIGHTS_DRIVE_IDS = {
    "migan_256_places2.pt": "1rEJ8uJszUOkP95hcq4Bbr-4M-dlyDeMa",
    "migan_512_places2.pt": "1D_YCuCgo20S2256sqpedsmENNm2WMtVY",
    "migan_256_ffhq.pt": "1uY0S-hWrRSv8WzwKMnoiDeWaZOk5i4g8",
}
WEIGHTS_DRIVE_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
WEIGHTS_SHA256 = {
    "migan_256_places2.pt": "8b82b2e82fc8e5a2e1f06827594aac1a5d66a9ca41e24199ddd969847788097f",
    "migan_512_places2.pt": "1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a",
    "migan_256_ffhq.pt": "3cab897cfcc3fcbe224510cedad06a634ae8bf5bd7120428a32fd7224f4237f4",
}
# size token -> checkpoint and input resolution. "256" and "512" are the Places2
# (scenery) weights at that resolution; "faces_256" is the FFHQ (faces)
# checkpoint, same 256 architecture, different weights, upstream ships it only
# at 256. The token is a size in the model.json sense (a distinct checkpoint),
# not necessarily a number.
CHECKPOINT_FOR_SIZE = {
    "256": "migan_256_places2.pt",
    "512": "migan_512_places2.pt",
    "faces_256": "migan_256_ffhq.pt",
}
RESOLUTION_FOR_SIZE = {"256": 256, "512": 512, "faces_256": 256}


class InpaintingWrapper(nn.Module):
    """The exported graph: plain image + mask in, inpainted image out.

    Folds upstream's `preprocess` / `postprocess` normalisation (see
    scripts/create_onnx_pipeline.py: MIGAN_Pipeline) into the graph so the app
    only feeds RGB in [0, 1] and a binary mask. Cropping to the hole's bounding
    box and blending the result back are NOT in here — they are data-dependent
    and belong in the app-side task.

    Inputs:
        image: [1, 3, H, W] float32, RGB in [0, 1]
        mask:  [1, 1, H, W] float32, 1.0 = known pixel (keep), 0.0 = hole
    Output:
        image: [1, 3, H, W] float32, RGB in [0, 1] (whole frame, hole filled)
    """

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img = image * 2.0 - 1.0
        x = torch.cat([mask - 0.5, img * mask], dim=1)
        out = self.generator(x)
        return (out * 0.5 + 0.5).clamp(0.0, 1.0)


class Downsample2dVulkan(nn.Module):
    """Vulkan-only stand-in for `Downsample2d`, numerically identical.

    Upstream's `Downsample2d` is a 4x4 depthwise conv with stride 2. The
    ExecuTorch Vulkan runtime routes a depthwise conv whose stride differs from
    its dilation to the generic `conv2d_dw_sned` shader, and the prebuilt
    react-native-executorch Vulkan library ships only the tiled 3x3 / 5x5
    variants of it, so the model fails to load ("Could not find ShaderInfo with
    name conv2d_dw_sned_half"). A 4x4 kernel fits neither tile. Running the
    same conv at stride 1 selects the plain `conv2d_dw` shader (shipped), and
    `avg_pool2d(kernel=1, stride=2)` keeps exactly the pixels the stride-2 conv
    would have produced: out[i] = y[2i] with the same padding of 1.
    """

    def __init__(self, downsample: "Downsample2d"):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=downsample.filter.in_channels,
            out_channels=downsample.filter.out_channels,
            kernel_size=4,
            groups=downsample.filter.groups,
            padding=1,
            bias=False,
            stride=1,
        )
        self.filter.weight = downsample.filter.weight

    def forward(self, x):
        y = self.filter(x)
        return F.avg_pool2d(y, kernel_size=1, stride=2)


def vulkanize(generator: Generator) -> Generator:
    """Swap every `Downsample2d` in `generator` for `Downsample2dVulkan` (in place)."""
    for module in generator.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, Downsample2d):
                setattr(module, name, Downsample2dVulkan(child))
    return generator
