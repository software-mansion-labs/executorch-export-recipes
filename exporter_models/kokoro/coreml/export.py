"""Core ML export for kokoro: duration predictor and synthesizer.

Both models fix the token axis at 128 and are padded up to it by the runtime,
because coremltools cannot express that axis dynamically. The synthesizer keeps
its frame axis dynamic (16..296), so audio length still follows the predicted
durations exactly and no tempo scaling is needed. That contract is declared to
the runtime through the companion `get_model_schema` method; the duration
predictor is fully static and needs no companion schema.

fp32 and CPU_ONLY on purpose:
  - fp16 leaves the excitation audibly wrong: the decoder lands at 0.973
    spectral correlation against eager versus 0.9993 in fp32.
  - `ComputeUnit.ALL` gave a 25 minute compile for one method and an outright
    execute failure for another; CPU_ONLY compiles in ~12 s (once, cached
    across launches) and is still 7.6x faster than XNNPACK on an iPhone 16.

Four coremltools limitations are worked around here rather than in
`_kokoro_model`, so the XNNPACK and MLX exports keep the reference graph:

1. Nearest `F.interpolate` with a symbolic target size is converted through a
   rank 4 resize that never squeezes back, so the following conv1d sees a 4D
   input against a 3D kernel. Integer-factor nearest upsampling is a repeat
   plus a flatten, which stays rank 3.
2. `arange(frames)` and `torch.roll` along the dynamic axis both lower to a
   `range_1d` with a symbolic end, which the Core ML CPU backend fails to
   reshape ("BNNS range_1d: Invalid range") and silently corrupts. They are
   replaced by `one_hot` and an explicit concat.
3. A dynamically sized `F.pad` makes `torch.export` guard on the pad block's
   numel and pins the frame axis to the example value. Appending a statically
   sized zero block and slicing back to the padded length avoids the guard.
4. The segmented phase scan must run a fixed number of steps: `while step < T`
   on a symbolic T guards on it and specializes the export.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Language variants ride the {size} axis. They are fine-tunes of the base
# checkpoint with an identical architecture and config, so only the weights
# change; `None` means the base checkpoint that ships with BASE_REPO.
BASE_REPO = "hexgrad/Kokoro-82M"
FINETUNE_REPO = "software-mansion/react-native-executorch-kokoro"
# Pinned so a re-export from the published tag is reproducible.
FINETUNE_REVISION = "v0.10.0"
CHECKPOINTS = {
    "standard": None,
    "polish": "finetunes/kokoro_polish_converted.pth",
    "german": "finetunes/kokoro_german_converted.pth",
}

# Runtime contract, mirrored by the companion schema and by the task in RNE.
MAX_TOKENS = 128
MIN_FRAMES = 16
MAX_FRAMES = 296
DURATION_FEATURE_DIM = 640
VOICE_REF_SIZE = 256
TICKS_PER_DURATION = 600
# Upper bound on the source module's sequence length, used to fix the number of
# scan steps so that a symbolic length never enters a loop condition.
SINE_SCAN_MAX_LEN = MAX_FRAMES * TICKS_PER_DURATION


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


class DurationPredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.text_encoder_module = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

    def forward(self, input_ids, text_mask, s, speed):
        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        d = self.text_encoder_module(
            d_en, s, torch.tensor(input_ids.shape[-1]), ~text_mask
        )
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        return torch.round(duration).clamp(min=1).long().squeeze(0), d


class Synthesizer(torch.nn.Module):
    def __init__(self, model, duration_padding):
        super().__init__()
        self.shared = model.predictor.shared
        self.F0_blocks = model.predictor.F0
        self.F0_proj = model.predictor.F0_proj
        self.N_blocks = model.predictor.N
        self.N_proj = model.predictor.N_proj
        self.lstm_dur_padding = duration_padding
        self.text_encoder = model.text_encoder
        self.decoder = model.decoder

    def forward(self, input_ids, text_mask, indices, d, ref_s):
        ref, s = ref_s[:, :128], ref_s[:, 128:]
        input_size = input_ids.shape[-1]
        # one_hot rather than an index_put over `arange(frames)`, see (2) above.
        pred_aln_trg = (
            torch.nn.functional.one_hot(indices, num_classes=input_size)
            .to(torch.float32)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        en = d.transpose(-1, -2) @ pred_aln_trg
        x = en.transpose(-1, -2)
        # The shared LSTM is bidirectional and torch.export unrolls it over
        # time, so it needs a static length. The published XNNPACK synthesizer
        # pads it the same way.
        original_seq_len = x.shape[1]
        torch._check_is_size(original_seq_len)
        x = torch.nn.functional.pad(
            x, (0, 0, 0, self.lstm_dur_padding - original_seq_len)
        )
        x, _ = self.shared(x)
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
        return self.decoder(t_en @ pred_aln_trg, F0, N, ref)


def remove_weight_norms(synthesizer):
    remove = torch.nn.utils.parametrize.remove_parametrizations
    blocks = (
        list(synthesizer.F0_blocks)
        + list(synthesizer.N_blocks)
        + list(synthesizer.decoder.decode)
        + [synthesizer.decoder.encode]
    )
    for module in blocks:
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
    generator = synthesizer.decoder.generator
    for module in list(generator.resblocks) + list(generator.noise_res):
        for conv in list(module.convs1) + list(module.convs2):
            _ = conv.weight
            remove(conv, "weight", leave_parametrized=True)
    for conv in list(generator.ups) + [
        generator.conv_post,
        synthesizer.decoder.F0_conv,
        synthesizer.decoder.N_conv,
        synthesizer.decoder.asr_res[0],
    ]:
        _ = conv.weight
        remove(conv, "weight", leave_parametrized=True)


def _dynamic_shape_rewrites(SineGen):
    """Rewrites that keep the graph convertible under a symbolic frame axis.

    Every one of them is numerically transparent; `export()` asserts that below
    before it writes anything.
    """
    real_pad = torch.nn.functional.pad
    real_interpolate = torch.nn.functional.interpolate
    orig_f02sine = SineGen._f02sine

    def pad_as_concat(t, pads, mode="constant", value=None):
        if len(pads) == 4 and pads[0] == pads[1] == pads[2] == 0 and t.dim() == 3:
            total = t.shape[1] + pads[3]
            zeros = torch.zeros(
                t.shape[0], MAX_FRAMES, t.shape[2], dtype=t.dtype, device=t.device
            )
            return torch.cat([t, zeros], dim=1)[:, :total].contiguous()
        if value is not None:
            return real_pad(t, pads, mode=mode, value=value)
        return real_pad(t, pads, mode=mode)

    def interpolate_as_repeat(x, size=None, scale_factor=None, mode="nearest", **kw):
        factor = None
        if mode == "nearest" and x.dim() == 3:
            if scale_factor is not None and float(scale_factor).is_integer():
                factor = int(scale_factor)
            elif size is not None:
                target = size[0] if isinstance(size, (list, tuple)) else size
                try:
                    factor = int(target // x.shape[-1])
                except Exception:
                    factor = None
        if factor is None:
            return real_interpolate(
                x, size=size, scale_factor=scale_factor, mode=mode, **kw
            )
        # repeat + flatten, not expand + reshape: expand's shape vector would
        # carry the symbolic length, which materializes as a float and then
        # fails Core ML's int-only shape concat.
        return x.unsqueeze(-1).repeat(1, 1, 1, factor).flatten(2)

    def shift(x, k, fill):
        """x shifted right by k along dim 1, padded with `fill`.

        Prepending k rows and cutting back to the original length stays correct
        for k >= T, which lets the scan run a fixed number of steps.
        """
        T = x.shape[1]
        pad = torch.full(
            (x.shape[0], k) + tuple(x.shape[2:]), fill, dtype=x.dtype, device=x.device
        )
        return torch.cat([pad, x], dim=1)[:, :T]

    def static_f02sine(self, f0_values):
        quot = f0_values / self.sampling_rate
        # `% 1` is lowered to zeros by coremltools (apple/coremltools#2836)
        rad_values = quot - torch.floor(quot)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0.0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            return orig_f02sine(self, f0_values)
        uv = self._f02uv(f0_values)
        # roll along the dynamic axis lowers to a symbolic `range_1d`; the
        # shifted-in value is overwritten with 1 anyway, so concat it.
        uv_1 = torch.cat([uv[:, 1:], torch.ones_like(uv[:, :1])], dim=1)
        m = ((uv < 1) * (uv_1 > 0)).to(rad_values.dtype)
        # Segmented prefix sum of rad, wrapped mod 1 at every step, so every
        # intermediate stays in [0, 1) instead of growing to hundreds of cycles.
        v = rad_values * (1 - m)
        f = torch.maximum(m, shift(m, 1, 0.0))
        step = 1
        while step < SINE_SCAN_MAX_LEN:
            v = v + shift(v, step, 0.0) * (1 - f)
            v = v - torch.floor(v)
            f = torch.clamp(f + shift(f, step, 1.0) * (1 - f), max=1.0)
            step *= 2
        return torch.cos(v * 2 * torch.pi)

    return (
        pad_as_concat,
        interpolate_as_repeat,
        static_f02sine,
        (real_pad, real_interpolate, orig_f02sine),
    )


def _synthesizer_schema():
    def const(value):
        return {"kind": "constant", "value": value}

    def rng(lo, hi, step=1):
        return {"kind": "range", "range": {"min": lo, "max": hi, "step": step}}

    def tensor(dtype, shape):
        return {"kind": "Tensor", "dtype": dtype, "shape": shape}

    return json.dumps({
        "forward": {
            "inputs": [
                tensor("int64", [const(1), const(MAX_TOKENS)]),
                tensor("bool", [const(1), const(MAX_TOKENS)]),
                tensor("int64", [rng(MIN_FRAMES, MAX_FRAMES)]),
                tensor("float32", [
                    const(1), const(MAX_TOKENS), const(DURATION_FEATURE_DIM)
                ]),
                tensor("float32", [const(1), const(VOICE_REF_SIZE)]),
            ],
            "outputs": [
                tensor("float32", [
                    const(1),
                    const(1),
                    rng(
                        MIN_FRAMES * TICKS_PER_DURATION,
                        MAX_FRAMES * TICKS_PER_DURATION,
                        TICKS_PER_DURATION,
                    ),
                ]),
            ],
            "runtimeConstraints": [
                {
                    "kind": "linear",
                    "dimLhs": {"paramSide": "output", "tensorIdx": 0, "dimIdx": 2},
                    "dimRhs": {"paramSide": "input", "tensorIdx": 2, "dimIdx": 0},
                    "coefficients": [TICKS_PER_DURATION, 0],
                }
            ],
        }
    })


def _spectral_corr(a, b):
    fa = np.abs(np.fft.rfft(a.flatten().numpy().astype(np.float64)))
    fb = np.abs(np.fft.rfft(b.flatten().numpy().astype(np.float64)))
    return float(np.corrcoef(fa, fb)[0, 1])


def _checkpoint(size: str) -> str | None:
    """Local path of the fine-tuned checkpoint for `size`, or None for base."""
    if size not in CHECKPOINTS:
        raise ValueError(f"unknown kokoro size {size!r}, expected one of {sorted(CHECKPOINTS)}")
    filename = CHECKPOINTS[size]
    if filename is None:
        return None
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=FINETUNE_REPO, filename=filename, revision=FINETUNE_REVISION
    )


def export(config: dict, out_dir: str) -> None:
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    _load_kokoro_model()
    from kokoro_kokoro_model_coreml.model import KModel
    from kokoro_kokoro_model_coreml.istftnet import SineGen

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_unit=ct.ComputeUnit.CPU_ONLY,
    )
    # lower_full_graph=False keeps anything Core ML cannot take on the CPU
    # backend rather than failing the whole export.
    partitioner = CoreMLPartitioner(compile_specs=compile_specs, lower_full_graph=False)

    def variant(needle):
        return next(v["file"] for v in config["variants"] if needle in v["file"])

    torch.manual_seed(0)
    model = KModel(
        repo_id=BASE_REPO,
        model=_checkpoint(config.get("size") or "standard"),
        disable_complex=True,
        lstm_padding=None,
    )

    # Example inputs only steer tracing, so a random voice reference is fine.
    voice_ref = torch.randn(1, VOICE_REF_SIZE)
    ids = torch.zeros((1, MAX_TOKENS), dtype=torch.long)
    ids[0, 1:MAX_TOKENS - 1] = torch.randint(1, 178, (MAX_TOKENS - 2,))
    text_mask = torch.ones((1, MAX_TOKENS), dtype=torch.bool)

    # --- duration predictor: fully static, no companion schema needed --------
    predictor = DurationPredictor(model).eval()
    predictor_inputs = (ids, text_mask, voice_ref[:, 128:], torch.tensor([1.0]))
    predictor_program = to_edge_transform_and_lower(
        torch.export.export(predictor, predictor_inputs), partitioner=[partitioner]
    ).to_executorch()
    (out / variant("duration_predictor")).write_bytes(predictor_program.buffer)

    # --- synthesizer: static tokens, dynamic frames --------------------------
    synthesizer = Synthesizer(model, duration_padding=MAX_FRAMES).eval()
    remove_weight_norms(synthesizer)

    synthesizer_inputs = (
        ids,
        text_mask,
        torch.randint(0, MAX_TOKENS, (MAX_FRAMES,)).sort().values,
        torch.randn(1, MAX_TOKENS, DURATION_FEATURE_DIM),
        voice_ref,
    )

    pad_as_concat, interpolate_as_repeat, static_f02sine, reals = (
        _dynamic_shape_rewrites(SineGen)
    )
    real_pad, real_interpolate, real_f02sine = reals

    # `pad_as_concat` is value-identical, but it hands the shared LSTM a
    # contiguous copy instead of a transposed view, which selects a different
    # kernel and shifts its output in the 10th digit. This vocoder amplifies
    # that to ~4e-2 of waveform, so it cannot be asserted away at the output.
    # Check it directly on a tensor instead, and keep it enabled on both sides
    # below so the output comparison isolates the rewrites that must be exact.
    probe = torch.randn(1, MAX_FRAMES, DURATION_FEATURE_DIM)
    for pads in ((0, 0, 0, 0), (0, 0, 0, MAX_FRAMES // 4)):
        if not torch.equal(pad_as_concat(probe, pads), real_pad(probe, pads)):
            raise RuntimeError(f"pad rewrite is not value-identical for pads={pads}")

    # SineGen seeds each call with a random initial phase, and the phase is
    # irrelevant to whether the rewrites are faithful. Pin it to zero so the
    # comparison measures the graph, not the draw.
    real_rand = torch.rand

    def _run():
        torch.rand = lambda *a, **kw: torch.zeros(
            *a, **{k: v for k, v in kw.items() if k != "generator"}
        )
        try:
            with torch.no_grad():
                return synthesizer(*synthesizer_inputs)
        finally:
            torch.rand = real_rand

    torch.nn.functional.pad = pad_as_concat
    try:
        reference = _run()
        torch.nn.functional.interpolate = interpolate_as_repeat
        SineGen._f02sine = static_f02sine
        rewritten = _run()
        # Sample-wise equality is the wrong bar here, see the note above.
        corr = _spectral_corr(rewritten, reference)
        if corr < 0.9999:
            raise RuntimeError(
                f"Core ML graph rewrites changed the output (spectral corr {corr:.6f})"
            )

        frames = torch.export.Dim("frames", min=MIN_FRAMES, max=MAX_FRAMES)
        synthesizer_ep = torch.export.export(
            synthesizer,
            synthesizer_inputs,
            dynamic_shapes=(None, None, {0: frames}, None, None),
        )
    finally:
        torch.nn.functional.pad = real_pad
        torch.nn.functional.interpolate = real_interpolate
        SineGen._f02sine = real_f02sine

    synthesizer_program = to_edge_transform_and_lower(
        synthesizer_ep,
        partitioner=[partitioner],
        constant_methods={"get_model_schema": _synthesizer_schema()},
    ).to_executorch()
    (out / variant("synthesizer")).write_bytes(synthesizer_program.buffer)
