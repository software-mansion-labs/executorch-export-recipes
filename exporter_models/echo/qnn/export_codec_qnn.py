"""Echo's codec decoder -> ExecuTorch QNN (Qualcomm HTP) at one static T. Not wired into
the harness (it has no qnn backend); see NOTES.md.

usage: python export_codec_qnn.py <out.pte> --precision fp16|a16w8 [--T 256] [--soc SM8850]
                                  [--calib-n 16]

  decode : codes i32[8, T] -> audio f32[T * 960]   (shorter clips pad the codes, trim the audio)

The HTP runs a height-1 [1, C, 1, L] tensor ~200x slower than a folded one, so time is folded
into rows ([1, C, R, W], t = r * W + w): 8 rows for the first two blocks, each conv seeing its
neighbour rows' edge columns as halos. From block 2 on, the activation is re-folded to 32 rows
and each block runs overlap-save: every row gets `margin` true neighbour columns per side, runs
the block on its own with plain zero padding, and keeps its centre; groups of 8 rows are
independent chains small enough for the HTP to keep on chip (fp16 128 -> 80 ms on SM8850).
Exact except the first ~3 ms and last ~13 ms, where the margins see computed values instead of
zero padding. Every transposed conv maps L to s * L, so row r stays time chunk r throughout.

HTP pitfalls this works around (QAIRT 2.47, SM8850):
- TransposeConv2d is wrong after 1-2 zero columns are padded or concatenated in front of it,
  and past ~8192 input columns: rows-alone pads 4 zero columns, wide inputs go in chunks.
- sin is not in QNN LayoutTransform's layout-agnostic list, and pow(x, 2) lowers to
  pow.Tensor_Tensor: both add NHWC<->NCHW permutes around every Snake; s * s and a patched list.
- a16w8: conv1 has dead output channels (|w| ~ 1e-40) whose 16-bit-eps weight scale the HTP
  miscomputes; weight scales are floored at the 8-bit eps.
a16w8 calibrates on real dev-set codes (--calib-n clips): 45 ms, ~27 dB waveform SNR to fp32,
Whisper WER unchanged on 21 held-out clips; fp16 is ~53 dB.
"""

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _echo import hub  # noqa: E402
from _echo.codec import load_codec  # noqa: E402

FOLD, TILE_FROM, TILE_FOLD, TILE_ROWS = 8, 2, 32, 8


class Snake(torch.nn.Module):
    """DAC Snake1d as x + c * sin^2(a x), fp16-safe: a = alpha, c = 1/alpha, except
    |alpha| < 1e-3 where a = 1e-3 * sign(alpha), c = alpha / a^2 (alpha * x^2 to O((a x)^2),
    the original's limit; 1/alpha itself overflows fp16 on 8 channels)."""

    SMALL = 1e-3

    def __init__(self, s):
        super().__init__()
        alpha = s.alpha.detach().clone()
        small = alpha.abs() < self.SMALL
        sign = torch.where(alpha < 0, -torch.ones_like(alpha), torch.ones_like(alpha))
        a = torch.where(small, self.SMALL * sign, alpha)
        self.register_buffer("a", a.unsqueeze(-1))  # [1, C, 1, 1] for folded activations
        self.register_buffer("c", torch.where(small, alpha / (a * a), (alpha + 1e-9).reciprocal()).unsqueeze(-1))

    def forward(self, x):
        s = torch.sin(self.a * x)
        return x + self.c * (s * s)


class Folded(torch.nn.Module):
    """The acoustic decoder on folded activations (module docstring)."""

    CONVT_MAX_W = 2560

    def __init__(self, ad):
        super().__init__()
        self.ad = ad
        # 4-D conv weights as constants (QNN wants them as parameters/buffers, and the
        # weight-norm parametrization would otherwise be recomputed in the graph).
        self.names = {}
        convs = (m for m in ad.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)))
        for i, m in enumerate(convs):
            self.register_buffer(f"w{i}", m.weight.detach().unsqueeze(2).clone())
            self.register_buffer(f"b{i}", m.bias.detach().clone())
            self.names[id(m)] = i

    def wb(self, m):
        i = self.names[id(m)]
        return getattr(self, f"w{i}"), getattr(self, f"b{i}")

    @staticmethod
    def halo(x, p):  # [1, C, R, W] -> [1, C, R, W + 2p], neighbour rows' edge columns, zeros at the ends
        z = x.new_zeros(x.shape[0], x.shape[1], 1, p)
        left = torch.cat([z, x[:, :, :-1, -p:]], dim=2)
        right = torch.cat([x[:, :, 1:, :p], z], dim=2)
        return torch.cat([left, x, right], dim=3)

    def conv(self, x, c, rows_alone=False):
        k, d = c.kernel_size[0], c.dilation[0]
        p = (k - 1) // 2 * d
        assert c.padding[0] == p and c.stride[0] == 1
        w, b = self.wb(c)
        if rows_alone:
            return F.conv2d(x, w, b, padding=(0, p), dilation=(1, d))
        return F.conv2d(self.halo(x, p) if p else x, w, b, dilation=(1, d))

    def conv_t(self, x, ct, rows_alone=False):
        s, p, k = ct.stride[0], ct.padding[0], ct.kernel_size[0]
        assert k <= 2 * s and p <= 2 * s
        w, b = self.wb(ct)
        if rows_alone:  # 4 zero columns: 1-2 hit the HTP TransposeConv2d bug, more only shift the output
            q = 4
            z = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], q)
            xh = torch.cat([z, x, z], dim=3)
        else:
            q = 1
            xh = self.halo(x, 1)
        n = x.shape[-1]
        bounds = list(range(0, n, self.CONVT_MAX_W)) + [n]
        ys = []
        for a, e in zip(bounds, bounds[1:]):
            y = F.conv_transpose2d(xh[..., a : e + 2 * q], w, b, stride=(1, s))
            ys.append(y[..., q * s + p : q * s + p + (e - a) * s])
        return ys[0] if len(ys) == 1 else torch.cat(ys, dim=3)

    def block(self, x, b, rows_alone=False):
        x = self.conv_t(b.snake1(x), b.conv_t1, rows_alone)
        for ru in (b.res_unit1, b.res_unit2, b.res_unit3):
            y = self.conv(ru.snake1(x), ru.conv1, rows_alone)
            x = x + self.conv(ru.snake2(y), ru.conv2, rows_alone)
        return x

    @staticmethod
    def margin(b):
        """Block-input columns a row needs from its neighbours: the residual units' reach at the
        output rate brought back through the stride, plus the transposed conv's."""
        reach = sum((c.kernel_size[0] - 1) // 2 * c.dilation[0]
                    for ru in (b.res_unit1, b.res_unit2, b.res_unit3) for c in (ru.conv1, ru.conv2))
        return -(-reach // b.conv_t1.stride[0]) + 2

    def block_tiled(self, x, b):
        m, s, W = self.margin(b), b.conv_t1.stride[0], x.shape[-1]
        xh = self.halo(x, m)
        outs = [self.block(xh[:, :, r : r + TILE_ROWS], b, rows_alone=True)[..., m * s : (m + W) * s]
                for r in range(0, x.shape[2], TILE_ROWS)]
        return torch.cat(outs, dim=2)

    def forward(self, x):  # [1, 256, T]
        ad = self.ad
        x = self.conv(x.view(1, x.shape[1], FOLD, -1), ad.conv1)
        for i, b in enumerate(ad.block):
            if i == TILE_FROM:
                x = x.reshape(1, x.shape[1], TILE_FOLD, -1)
            x = self.block_tiled(x, b) if i >= TILE_FROM else self.block(x, b)
        return ad.tanh(self.conv(ad.snake1(x), ad.conv2))  # tanh is nn.Identity in this codec


class Decoder(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.q, self.fc2 = codec.quantizer, codec.fc2
        self.dec = Folded(codec.acoustic_decoder)

    def forward(self, codes):  # int32 [8, T]
        codes = codes.long()
        qs = self.q.quantizers
        x = qs[0].project_out(qs[0].codebook.decode(codes[0]))
        for i in range(1, len(qs)):
            x = x + qs[i].project_out(qs[i].codebook.decode(codes[i]))
        return self.dec(self.fc2(x).view(1, -1, 256).permute(0, 2, 1)).reshape(-1)


def swap_snake(m):
    for name, child in m.named_children():
        if type(child).__name__ == "Snake1d":
            setattr(m, name, Snake(child))
        else:
            swap_snake(child)


def calib_codes(n, T):
    """n dev-set code sets as int32 [8, T] (short clips repeat)."""
    from calibrate import dev_set

    tmp = tempfile.TemporaryDirectory()
    rows, _ = dev_set(Path(tmp.name))
    random.Random(0).shuffle(rows)
    out = []
    for r in rows[:n]:
        c = np.load(Path(tmp.name) / r["name"])
        c = c if c.shape[0] == 8 else c.T
        c = np.tile(c, (1, -(-T // c.shape[1])))[:, :T]
        out.append(torch.from_numpy(c.astype(np.int32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--precision", required=True, choices=["fp16", "a16w8"])
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--soc", default="SM8850")
    ap.add_argument("--calib-n", type=int, default=16, help="a16w8: dev-set clips to calibrate on")
    args = ap.parse_args()
    assert args.T % TILE_FOLD == 0
    torch.set_grad_enabled(False)

    codec = load_codec(hub.codec())
    ex = (torch.randint(0, 1024, (8, args.T), dtype=torch.int32),)
    ref = codec.decode(ex[0].long()[None]).audio_values.reshape(-1)
    swap_snake(codec.acoustic_decoder)
    dec = Decoder(codec).eval()
    d = (dec(*ex) - ref)[2400:-4800].abs().max()  # tiles: the two sequence ends differ (docstring)
    print(f"folded vs original codec (without the first 0.1 s / last 0.2 s): max abs {float(d):.2e}")
    assert d < 1e-3

    from executorch.backends.qualcomm._passes.layout_transform import LayoutTransform
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    LayoutTransform.layout_agnostic_ops.add(exir_ops.edge.aten.sin.default)
    quantized = args.precision == "a16w8"
    mod = dec
    if quantized:
        from executorch.backends.qualcomm.export_utils import make_quantizer
        from executorch.backends.qualcomm.quantizer.qconfig import DEFAULT_EPS_8BIT
        from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
        from torchao.quantization.pt2e.observer import MinMaxObserver
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        prepared = prepare_pt2e(
            torch.export.export(dec, ex, strict=False).module(),
            make_quantizer(quant_dtype=QuantDtype.use_16a8w, soc_model=args.soc, act_observer=MinMaxObserver,
                           per_channel_linear=True, eps=DEFAULT_EPS_8BIT),
        )
        for codes in calib_codes(args.calib_n, args.T):
            prepared(codes)
        mod = convert_pt2e(prepared)

    specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized),
    )
    t0 = time.time()
    edge = to_edge_transform_and_lower_to_qnn({"decode": mod}, {"decode": ex}, specs)
    print(f"lowering ok ({time.time() - t0:.0f}s)")
    prog = edge.to_executorch(config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
        segment_alignment=get_qnn_context_binary_alignment(),
    ))
    Path(args.out).write_bytes(prog.buffer)
    print(f"wrote {args.out} ({len(prog.buffer) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
