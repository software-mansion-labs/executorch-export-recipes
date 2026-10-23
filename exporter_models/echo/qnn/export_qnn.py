"""Echo -> ExecuTorch QNN (Qualcomm HTP), one static (P, T) bucket. Not wired
into the harness (it has no qnn backend); see NOTES.md for the environment.

usage: python export_qnn.py <out.pte> --precision fp16|a16w8 [--calib calib.pt]
                            [--P 320] [--T 256] [--soc SM8850]

  encode_prefix : (text i32[P], audio i32[8, P], is_audio f32[P], pos i32[P], key_bias f32[P])
                  -> 48 x f32[6, P, 128]  (+ hidden f32[P, 1024] for a16w8)
  denoise       : (codes i32[8, T], pos i32[T], bias f32[P], self_bias f32[T], 48 x f32[6, P, 128])
                  -> log_probs f32[8, T, 1024]

The HTP has no int64 tensors, so ids and positions are int32. The caller pads
the prefix to P and the target to T: key_bias / bias carry -1e4 on padded
prefix keys and self_bias on padded target keys; padded rows are garbage.
fp16 runs the fp32 checkpoint; a16w8 runs the w8a16 QAT checkpoint through
PT2E calibration (calibrate.py) and also returns the last hidden state to
dodge pytorch/executorch#24125. Both methods share one weight set on the HTP.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _echo import hub  # noqa: E402
from _echo.model import Echo, RMSNorm  # noqa: E402


class Prefix(torch.nn.Module):
    def __init__(self, m, return_hidden):
        super().__init__()
        self.m, self.return_hidden = m, return_hidden

    def forward(self, text, audio, is_audio, pos, key_bias):
        return self.m.encode_prefix(text.long(), audio.long(), is_audio, pos.long(), key_bias,
                                    return_hidden=self.return_hidden)


class Denoise(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes, pos, bias, self_bias, kv):
        return self.m.denoise(codes.long(), pos.long(), bias, *kv, self_bias=self_bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--precision", required=True, choices=["fp16", "a16w8"])
    ap.add_argument("--calib", help="calibrate.py output at the same bucket (required for a16w8)")
    ap.add_argument("--P", type=int, default=320)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--soc", default="SM8850")
    args = ap.parse_args()
    quantized = args.precision == "a16w8"
    if quantized and not args.calib:
        ap.error("a16w8 needs --calib (calibrate.py)")
    torch.set_grad_enabled(False)

    # QNN's RmsNorm op; the decomposed x * x overflows fp16 (|x| ~ 1.1e4).
    RMSNorm.mode = "native"
    model = Echo.from_pretrained(hub.checkpoint("w8a16" if quantized else "fp32"))
    n_kv = 2 * len(model.llm.layers)
    kv_heads, head_dim = model.cfg["num_key_value_heads"], model.cfg["head_dim"]
    P, T, i32 = args.P, args.T, torch.int32
    pre_in = (torch.zeros(P, dtype=i32), torch.zeros(8, P, dtype=i32), torch.zeros(P), torch.arange(P, dtype=i32),
              torch.zeros(P))
    den_in = (torch.full((8, T), 1024, dtype=i32), torch.arange(P, P + T, dtype=i32), torch.zeros(P), torch.zeros(T),
              tuple(torch.zeros(kv_heads, P, head_dim) for _ in range(n_kv)))
    methods = {"encode_prefix": (Prefix(model, quantized).eval(), pre_in), "denoise": (Denoise(model).eval(), den_in)}

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=generate_htp_compiler_spec(use_fp16=not quantized, use_weight_sharing=True),
    )

    modules, inputs = {}, {}
    for name, (mod, ex) in methods.items():
        if quantized:
            from executorch.backends.qualcomm.export_utils import make_quantizer
            from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
            from torchao.quantization.pt2e.observer import MinMaxObserver
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            prepared = prepare_pt2e(
                torch.export.export(mod, ex, strict=False).module(),
                make_quantizer(quant_dtype=QuantDtype.use_16a8w, soc_model=args.soc, act_observer=MinMaxObserver,
                               per_channel_linear=True),
            )
            for pre, den in torch.load(args.calib):
                prepared(*(pre if name == "encode_prefix" else den))
            mod = convert_pt2e(prepared)
        modules[name], inputs[name] = mod, ex

    t0 = time.time()
    edge = to_edge_transform_and_lower_to_qnn(modules, inputs, specs)
    print(f"lowering ok ({time.time() - t0:.0f}s)")
    prog = edge.to_executorch(config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
        segment_alignment=get_qnn_context_binary_alignment(),
    ))
    Path(args.out).write_bytes(prog.buffer)
    print(f"wrote {args.out} ({len(prog.buffer) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
