"""EasyOCR model architecture — the single source of truth for CRAFT + CRNN.

Pure model code (no ExecuTorch / backend imports): the VGG-backbone CRAFT detector
(static `size=` and dynamic `scale_factor=2` variants), the per-language CRNN
recognizer, and the scan-based BiLSTM that keeps the recognizer's time dim dynamic
through torch.export.

  CRAFT  [1,3,H,W]  -> ([1,H/2,W/2,2] region+affinity, [1,32,H/2,W/2] feature)
  CRNN   [1,1,64,W] -> [1, W/4-1, num_class]   (one head per language)

Dynamic enablers (both verified lossless):
  - CRAFTDyn: U-Net `F.interpolate(size=...)` (symbolic size = the CoreML blocker)
    rewritten to `scale_factor=2` — exact for inputs that are multiples of 32.
  - ScanBiLSTM: nn.LSTM specializes the time dim, so the BiLSTM is reimplemented
    via a single torch `scan` (both directions, cloned outputs) -> T stays symbolic.
"""
from collections import OrderedDict, namedtuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import models
from packaging import version
from torch._higher_order_ops.scan import scan

_sig, _th = torch.sigmoid, torch.tanh


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None: m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1); m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01); m.bias.data.zero_()


class vgg16_bn(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            feats = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None).features
        else:
            feats = models.vgg16_bn(pretrained=pretrained).features
        self.slice1, self.slice2, self.slice3, self.slice4 = (nn.Sequential() for _ in range(4))
        for x in range(12): self.slice1.add_module(str(x), feats[x])
        for x in range(12, 19): self.slice2.add_module(str(x), feats[x])
        for x in range(19, 29): self.slice3.add_module(str(x), feats[x])
        for x in range(29, 39): self.slice4.add_module(str(x), feats[x])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1))
        init_weights(self.slice5.modules())

    def forward(self, X):
        h = self.slice1(X); r22 = h
        h = self.slice2(h); r33 = h
        h = self.slice3(h); r43 = h
        h = self.slice4(h); r53 = h
        h = self.slice5(h)
        return namedtuple("V", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])(
            h, r53, r43, r33, r22)


def copyStateDict(state_dict):
    start = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return OrderedDict((".".join(k.split(".")[start:]), v) for k, v in state_dict.items())


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, 1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)


class CRAFT(nn.Module):
    """Static CRAFT (size= interpolation). Use CRAFTDyn for dynamic/enumerated."""
    def __init__(self):
        super().__init__()
        self.basenet = vgg16_bn(False, False)
        self.upconv1 = double_conv(1024, 512, 256); self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64);  self.upconv4 = double_conv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 2, 1))
        for m in (self.upconv1, self.upconv2, self.upconv3, self.upconv4, self.conv_cls):
            init_weights(m.modules())

    def forward(self, x):
        s = self.basenet(x)
        y = self.upconv1(torch.cat([s[0], s[1]], 1))
        y = F.interpolate(y, size=s[2].size()[2:], mode="bilinear", align_corners=False)
        y = self.upconv2(torch.cat([y, s[2]], 1))
        y = F.interpolate(y, size=s[3].size()[2:], mode="bilinear", align_corners=False)
        y = self.upconv3(torch.cat([y, s[3]], 1))
        y = F.interpolate(y, size=s[4].size()[2:], mode="bilinear", align_corners=False)
        feature = self.upconv4(torch.cat([y, s[4]], 1))
        return self.conv_cls(feature).permute(0, 2, 3, 1), feature


class CRAFTDyn(CRAFT):
    """CRAFT with scale_factor=2 upsampling (no symbolic size= dep) — exact for x32."""
    def forward(self, x):
        s = self.basenet(x)
        y = self.upconv1(torch.cat([s[0], s[1]], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.upconv2(torch.cat([y, s[2]], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.upconv3(torch.cat([y, s[3]], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        feature = self.upconv4(torch.cat([y, s[4]], 1))
        return self.conv_cls(feature).permute(0, 2, 3, 1), feature


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class ScanBiLSTM(nn.Module):
    """nn.LSTM (bidirectional) via a SINGLE torch scan -> dynamic time dim survives
    to_edge. One scan call (fwd + pre-flipped bwd as two xs leaves, 4-tensor carry)
    dodges the dynamo double-scan recompile-guard bug; per-step output is cloned to
    avoid carry/output aliasing. Runs portable (scan isn't a delegate op), lossless."""
    def __init__(self, lstm):
        super().__init__(); self.H = lstm.hidden_size
        for d, suf in (("f", ""), ("b", "_reverse")):
            self.register_parameter(f"wih_{d}", nn.Parameter(getattr(lstm, f"weight_ih_l0{suf}").data.clone()))
            self.register_parameter(f"whh_{d}", nn.Parameter(getattr(lstm, f"weight_hh_l0{suf}").data.clone()))
            self.register_parameter(f"bih_{d}", nn.Parameter(getattr(lstm, f"bias_ih_l0{suf}").data.clone()))
            self.register_parameter(f"bhh_{d}", nn.Parameter(getattr(lstm, f"bias_hh_l0{suf}").data.clone()))
    def forward(self, x):                          # [B,T,I] -> [B,T,2H]
        H, B = self.H, x.shape[0]
        xt = x.transpose(0, 1); xr = torch.flip(xt, [0])
        init = [torch.zeros(B, H, dtype=x.dtype) for _ in range(4)]
        def cell(c, xs):
            hf, cf, hb, cb = c; xf, xb = xs[0], xs[1]
            gf = xf @ self.wih_f.t() + self.bih_f + hf @ self.whh_f.t() + self.bhh_f
            cf2 = _sig(gf[:, H:2*H]) * cf + _sig(gf[:, :H]) * _th(gf[:, 2*H:3*H]); hf2 = _sig(gf[:, 3*H:4*H]) * _th(cf2)
            gb = xb @ self.wih_b.t() + self.bih_b + hb @ self.whh_b.t() + self.bhh_b
            cb2 = _sig(gb[:, H:2*H]) * cb + _sig(gb[:, :H]) * _th(gb[:, 2*H:3*H]); hb2 = _sig(gb[:, 3*H:4*H]) * _th(cb2)
            return [hf2, cf2, hb2, cb2], [hf2.clone(), hb2.clone()]
        _, ys = scan(cell, init, [xt, xr], dim=0)
        return torch.cat([ys[0], torch.flip(ys[1], [0])], -1).transpose(0, 1)


class ScanBiL(nn.Module):
    def __init__(self, orig): super().__init__(); self.s = ScanBiLSTM(orig.rnn); self.linear = orig.linear
    def forward(self, x): return self.linear(self.s(x))


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super().__init__()
        oc = [output_channel // 8, output_channel // 4, output_channel // 2, output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, oc[0], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[0], oc[1], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[1], oc[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(oc[2], oc[2], 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[2], oc[3], 3, 1, 1, bias=False), nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.Conv2d(oc[3], oc[3], 3, 1, 1, bias=False), nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[3], oc[3], 2, 1, 0), nn.ReLU(True))
    def forward(self, x): return self.ConvNet(x)


class CRNNModel(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super().__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.AvgPool = nn.AvgPool2d(kernel_size=(1, 3), stride=1)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.Prediction = nn.Linear(hidden_size, num_class)
    def forward(self, x):
        v = self.FeatureExtraction(x)
        v = self.AvgPool(v.permute(0, 3, 1, 2)).squeeze(3)
        return self.Prediction(self.SequenceModeling(v).contiguous())


def normalizeMeanVariance(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = img.copy().astype(np.float32)
    img -= np.array([m * 255.0 for m in mean], dtype=np.float32)
    img /= np.array([v * 255.0 for v in variance], dtype=np.float32)
    return img
