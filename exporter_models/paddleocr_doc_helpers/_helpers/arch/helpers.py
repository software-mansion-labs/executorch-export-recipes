"""Supporting helpers (orientation / dewarp / table) as bare-tensor modules.

Each HF model returns a `*ModelOutput`; the wrapper exposes the tensor we export.
`TDec` is one bit-exact AR step of SLANetSLAHead's decode loop (the loop itself is
data-dependent, so it runs in C++).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (PPLCNetForImageClassification, UVDocModel,
                          SLANetForTableRecognition)

ORI_ID = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
DEWARP_ID = "PaddlePaddle/UVDoc_safetensors"
TABLE_ID = "PaddlePaddle/SLANet_plus_safetensors"
OUT_CH, HID = 50, 256
METHODS = ["orientation", "dewarp", "table_encode", "table_decode_step"]

# int8-calibration normalization per method: PP-LCNet + SLANet backbone are
# ImageNet-normalized, UVDoc is plain /255. The recurrent decode has no image input.
CALIB_NORM = {"orientation": "imagenet", "dewarp": "div255", "table_encode": "imagenet"}


class _LastHidden(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).last_hidden_state


class TEnc(nn.Module):
    def __init__(self, m): super().__init__(); self.backbone = m.backbone
    def forward(self, x): return self.backbone(x).last_hidden_state


class TDec(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.cell = m.head.structure_attention_cell
        self.gen = m.head.structure_generator

    def forward(self, features, prev_hidden, prev_onehot):
        bhp = self.cell.input_to_hidden(features)
        php = self.cell.hidden_to_hidden(prev_hidden).unsqueeze(1)
        s = self.cell.score(torch.tanh(bhp + php))
        w = F.softmax(s, dim=1, dtype=torch.float32).to(s.dtype).transpose(1, 2)
        ctx = torch.matmul(w, features).squeeze(1)
        new_hidden = self.cell.rnn(torch.cat([ctx, prev_onehot], 1), prev_hidden)
        return F.softmax(self.gen(new_hidden), dim=-1), new_hidden


def load_modules():
    ori = PPLCNetForImageClassification.from_pretrained(ORI_ID).eval()
    dw = UVDocModel.from_pretrained(DEWARP_ID).eval()
    tbl = SLANetForTableRecognition.from_pretrained(TABLE_ID).eval()
    return {"orientation": _LastHidden(ori), "dewarp": _LastHidden(dw),
            "table_encode": TEnc(tbl), "table_decode_step": TDec(tbl)}


def example_inputs(name):
    return {
        "orientation":       lambda: (torch.randn(1, 3, 224, 224),),
        "dewarp":            lambda: (torch.randn(1, 3, 712, 488),),
        "table_encode":      lambda: (torch.randn(1, 3, 488, 488),),
        "table_decode_step": lambda: (torch.randn(1, HID, 96), torch.zeros(1, HID),
                                      torch.zeros(1, OUT_CH)),
    }[name]()
