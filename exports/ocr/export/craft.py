import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import models
from collections import OrderedDict
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from packaging import version
import cv2
import numpy as np
from collections import namedtuple


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:  # torchvision.__version__ < 0.13
            models.vgg.model_urls["vgg16_bn"] = models.vgg.model_urls[
                "vgg16_bn"
            ].replace("https://", "http://")
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"]
        )
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """Base network"""
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    img = in_img.copy().astype(np.float32)
    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def export_model_for_size(height, width):
    """Export model for a specific height and width."""
    print(f"\n{'='*60}")
    print(f"Exporting CRAFT model for size {height}x{width}")
    print(f"{'='*60}")

    # Create and load model
    model = CRAFT()
    model.load_state_dict(
        copyStateDict(
            torch.load("craft_mlt_25k.pth", map_location="cpu", weights_only=False)
        )
    )
    model.eval()

    # Create sample inputs for this size
    sample_inputs = (torch.randn(1, 3, height, width),)
    output = model(sample_inputs[0])

    # Setup quantization config
    qparams = get_symmetric_quantization_config(
        is_per_channel=False,
        is_qat=False,
    )
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)

    # Export non-quantized model
    print(f"Exporting non-quantized model for {height}x{width}...")
    exported_program = torch.export.export(model, sample_inputs)
    et_program_non_quant = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    non_quant_filename = f"xnnpack_craft_{width}.pte"
    with open(non_quant_filename, "wb") as f:
        f.write(et_program_non_quant.buffer)
    print(f"Non-quantized model saved to {non_quant_filename}")

    # Export quantized model with calibration
    print(f"\nExporting quantized model for {height}x{width}...")
    training_ep = exported_program.module()
    prepared_model = prepare_pt2e(training_ep, quantizer)

    # Calibrate with real images
    print("Calibrating with real image data...")
    calibration_images = [
        "./calibration_images/1.png",
        "./calibration_images/2.png",
        "./calibration_images/3.png",
        "./calibration_images/4.png",
    ]
    calibrated = 0

    with torch.no_grad():
        for img_path in calibration_images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                resized_img = cv2.resize(
                    img, (width, height), interpolation=cv2.INTER_LINEAR
                )
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                x = np.transpose(normalizeMeanVariance(resized_img), (2, 0, 1))
                cal_sample = torch.from_numpy(np.array([x]))
                prepared_model(cal_sample)
                calibrated += 1

        if calibrated > 0:
            print(f"  Calibrated with {calibrated} real images")
        else:
            # Fallback to synthetic data
            print("  Using synthetic calibration data")
            for i in range(10):
                cal_sample = torch.randn(1, 3, height, width) * 0.5
                prepared_model(cal_sample)

    quantized_model = convert_pt2e(prepared_model)

    et_program = to_edge_transform_and_lower(
        torch.export.export(quantized_model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    quant_filename = f"xnnpack_craft_{width}_quantized.pte"
    with open(quant_filename, "wb") as f:
        f.write(et_program.buffer)
    print(f"Quantized model saved to {quant_filename}")
    print(f"✓ Completed export for {height}x{width}\n")


# Export models for all three sizes
sizes = [(1280, 1280), (1280, 320), (800, 800)]

for height, width in sizes:
    export_model_for_size(height, width)
