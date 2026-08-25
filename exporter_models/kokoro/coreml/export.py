"""Core ML export for kokoro.

Only the synthesizer is exported here. The duration predictor stays on XNNPACK:
the shipping contract is a single dynamic `forward` (see the v0.10.0 artifact),
and coremltools cannot express it today, because dynamic `F.pad` on a rank > 1
tensor is unsupported and the `torch.cat` workaround makes `torch.export` derive
a non-equality constraint on the token dim. The synthesizer is also where the
time goes: on an iPhone 16 it is 627 ms on Core ML against 4741 ms on XNNPACK
(warm, 7.4 s of audio), while the duration predictor is 233 ms on XNNPACK.

fp32 only, and CPU_ONLY on purpose:
  - fp16 leaves the excitation audibly wrong. The phase accumulator is fine now
    that it is wrapped (see `_segmented_wrapped_phase`), but the decoder in fp16
    still lands at 0.973 spectral correlation against eager versus 0.9993 in
    fp32.
  - `ComputeUnit.ALL` on this model gave a 25 minute compile for one method and
    an outright execute failure for another; CPU_ONLY compiles in ~12 s (once,
    cached across launches) and is the 7.6x win above.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_kokoro_model():
    pkg_init = _MODEL_DIR / "_kokoro_model" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "kokoro_kokoro_model_coreml", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_kokoro_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kokoro_kokoro_model_coreml"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> None:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    _load_kokoro_model()
    from kokoro_kokoro_model_coreml.model import KModel

    class Synthesizer(torch.nn.Module):
        def __init__(self, model, duration_padding=None):
            super().__init__()
            self.shared = model.predictor.shared
            self.F0_blocks = model.predictor.F0
            self.F0_proj = model.predictor.F0_proj
            self.N_blocks = model.predictor.N
            self.N_proj = model.predictor.N_proj
            self.lstm_dur_padding = duration_padding
            self.text_encoder = model.text_encoder
            self.decoder = model.decoder

        def forward(self, input_ids: torch.LongTensor,
                    text_mask: torch.BoolTensor,
                    indices: torch.LongTensor,
                    d: torch.FloatTensor,
                    ref_s: torch.FloatTensor):
            ref = ref_s[:, :128]
            s = ref_s[:, 128:]

            input_size = input_ids.shape[-1]
            # one_hot instead of `pred_aln_trg[indices, arange] = 1.0`: same
            # alignment matrix without a data-dependent index_put
            pred_aln_trg = (
                torch.nn.functional.one_hot(indices, num_classes=input_size)
                .to(torch.float32)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            en = d.transpose(-1, -2) @ pred_aln_trg

            x = en.transpose(-1, -2)
            torch._check_is_size(x.shape[1])

            if self.lstm_dur_padding is not None:
                original_seq_len = x.shape[1]
                torch._check_is_size(original_seq_len)
                x = torch.nn.functional.pad(
                    x, (0, 0, 0, self.lstm_dur_padding - original_seq_len)
                )

            x, _ = self.shared(x)

            if self.lstm_dur_padding is not None:
                x = x[:, :original_seq_len, :]

            F0 = x.transpose(-1, -2)
            for block in self.F0_blocks:
                F0 = block(F0, s)
            F0 = self.F0_proj(F0).squeeze(1)

            N = x.transpose(-1, -2)
            for block in self.N_blocks:
                N = block(N, s)
            N = self.N_proj(N).squeeze(1)

            t_en = self.text_encoder(input_ids, ~text_mask)
            asr = t_en @ pred_aln_trg

            return self.decoder(asr, F0, N, ref)

    def remove_weight_norms(synthesizer):
        remove = torch.nn.utils.parametrize.remove_parametrizations
        modules = list(synthesizer.F0_blocks) + list(synthesizer.N_blocks)
        for module in modules:
            convs = [module.conv1, module.conv2]
            if hasattr(module.pool, "weight"):
                convs.append(module.pool)
            if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
                convs.append(module.conv1x1)
            for conv in convs:
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)

        for module in synthesizer.text_encoder.cnn:
            conv = module[0]
            _ = conv.weight
            remove(conv, "weight", leave_parametrized=True)

        for module in synthesizer.decoder.generator.resblocks:
            for conv in list(module.convs1) + list(module.convs2):
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for module in synthesizer.decoder.generator.noise_res:
            for conv in list(module.convs1) + list(module.convs2):
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for conv_t in synthesizer.decoder.generator.ups:
            _ = conv_t.weight
            remove(conv_t, "weight", leave_parametrized=True)
        _ = synthesizer.decoder.generator.conv_post.weight
        remove(synthesizer.decoder.generator.conv_post, "weight", leave_parametrized=True)

        modules = list(synthesizer.decoder.decode) + [synthesizer.decoder.encode]
        for module in modules:
            convs = [module.conv1, module.conv2]
            if hasattr(module.pool, "weight"):
                convs.append(module.pool)
            if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
                convs.append(module.conv1x1)
            for conv in convs:
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for conv in [synthesizer.decoder.F0_conv, synthesizer.decoder.N_conv,
                     synthesizer.decoder.asr_res[0]]:
            _ = conv.weight
            remove(conv, "weight", leave_parametrized=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    syn_inputs = (
        torch.randint(0, 178, size=(1, 128)),
        torch.ones((1, 128), dtype=torch.bool),
        torch.randint(0, 128, (296,), dtype=torch.long).sort().values,
        torch.randn(size=(1, 128, 640)),
        torch.randn(size=(1, 256)),
    )

    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=None)
    synthesizer = Synthesizer(model, duration_padding=None)
    synthesizer.eval()
    remove_weight_norms(synthesizer)

    syn_ep = torch.export.export(synthesizer, syn_inputs)

    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_unit=ct.ComputeUnit.CPU_ONLY,
    )
    # lower_full_graph=False keeps anything Core ML cannot take on the CPU
    # backend rather than failing the whole export.
    partitioner = CoreMLPartitioner(compile_specs=compile_specs, lower_full_graph=False)

    syn_program = to_edge_transform_and_lower(
        syn_ep, partitioner=[partitioner]
    ).to_executorch()

    target = next(v["file"] for v in config["variants"] if "synthesizer" in v["file"])
    (out / target).write_bytes(syn_program.buffer)
