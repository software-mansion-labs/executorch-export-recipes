"""Extract weights from the Supertonic 3 ONNX files into PyTorch state dicts.

Usage:
    python -m supertonic_torch.convert --onnx-dir onnx --out checkpoints/supertonic3.pt

Most initializers keep their original PyTorch names in the ONNX graphs
(``tts.dp....``, ``vector_estimator.tts.ttl....``); those are mapped by
stripping the prefixes. Linear weights fused by the exporter into anonymous
``onnx::MatMul_*`` initializers are recovered from the node names of the
MatMul ops consuming them (and transposed back to [out, in]).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def _load_initializers(model):
    from onnx import numpy_helper

    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


def _anon_linear_weights(model, node_prefix_strip):
    """Map anonymous MatMul weight initializers to state-dict names.

    A node like ``/vector_estimator/vector_field/main_blocks.5/attention/
    W_key/linear/MatMul`` with an ``onnx::MatMul_*`` initializer input yields
    ``vector_field.main_blocks.5.attention.W_key.linear.weight`` ([out, in]).
    """
    inits = _load_initializers(model)
    out = {}
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        w_name = node.input[1]
        if w_name not in inits or not w_name.startswith("onnx::"):
            continue
        m = re.match(r"^(.*)/linear/MatMul", node.name)
        if not m:
            continue
        path = m.group(1)
        if node_prefix_strip and path.startswith(node_prefix_strip):
            path = path[len(node_prefix_strip):]
        key = path.strip("/").replace("/", ".") + ".linear.weight"
        out[key] = np.ascontiguousarray(inits[w_name].T)  # [in,out] -> [out,in]
    return out


def _strip_named(inits, prefix, rename=None):
    out = {}
    for name, arr in inits.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        if rename:
            key = rename(key)
        out[key] = arr
    return out


def convert_text_encoder(path):
    import onnx
    model = onnx.load(path)
    inits = _load_initializers(model)
    sd = _strip_named(inits, "tts.ttl.")
    sd.update(_anon_linear_weights(model, node_prefix_strip=""))
    return sd


def convert_duration_predictor(path):
    import onnx
    model = onnx.load(path)
    inits = _load_initializers(model)
    return _strip_named(inits, "tts.dp.")


def convert_vector_estimator(path):
    import onnx
    model = onnx.load(path)
    inits = _load_initializers(model)
    sd = _strip_named(inits, "vector_estimator.tts.ttl.")
    sd.update(_anon_linear_weights(model, node_prefix_strip="/vector_estimator/"))
    # Learned style keys (constant-folded by the exporter).
    sd["style_key"] = inits["/vector_estimator/Expand_output_0"]
    # Sinusoidal time-embedding frequencies.
    sd["vector_field.time_encoder.freqs"] = inits[
        "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0"
    ]
    # Rotary params are stored only on block 3 in the ONNX graph but shared
    # by blocks 9/15/21 (identical modules); replicate them.
    for blk in (9, 15, 21):
        for p in ("theta", "increments"):
            sd[f"vector_field.main_blocks.{blk}.attn.{p}"] = sd[
                f"vector_field.main_blocks.3.attn.{p}"
            ]
    return sd


def convert_vocoder(path):
    import onnx
    model = onnx.load(path)
    inits = _load_initializers(model)
    sd = _strip_named(inits, "tts.ae.", rename=lambda k: "ae." + k)
    sd["ttl_scale"] = inits["tts.ttl.normalizer.scale"]
    # Anonymous embed conv + PReLU slope, recovered from the consuming nodes.
    for node in model.graph.node:
        if node.op_type == "Conv" and node.name == "/decoder/embed/net/Conv":
            sd["ae.decoder.embed.net.weight"] = inits[node.input[1]]
            sd["ae.decoder.embed.net.bias"] = inits[node.input[2]]
        if node.op_type == "PRelu":
            sd["ae.decoder.head.act.weight"] = inits[node.input[1]].reshape(1)
    return sd


def to_torch(sd):
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


def convert_all(onnx_dir):
    onnx_dir = Path(onnx_dir)
    return {
        "text_encoder": to_torch(convert_text_encoder(onnx_dir / "text_encoder.onnx")),
        "duration_predictor": to_torch(
            convert_duration_predictor(onnx_dir / "duration_predictor.onnx")
        ),
        "vector_estimator": to_torch(
            convert_vector_estimator(onnx_dir / "vector_estimator.onnx")
        ),
        "vocoder": to_torch(convert_vocoder(onnx_dir / "vocoder.onnx")),
    }


def load_models(checkpoint_path=None, onnx_dir=None, device="cpu"):
    """Build the four PyTorch models and load extracted weights."""
    from .models import (
        DurationPredictorModel,
        TextEncoderModel,
        VectorEstimatorModel,
        VocoderModel,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        ckpt = convert_all(onnx_dir)

    models = {
        "text_encoder": TextEncoderModel(),
        "duration_predictor": DurationPredictorModel(),
        "vector_estimator": VectorEstimatorModel(),
        "vocoder": VocoderModel(),
    }
    for name, model in models.items():
        sd = ckpt[name]
        if name == "vocoder":  # BatchNorm bookkeeping buffer absent in ONNX
            sd = dict(sd)
            sd.setdefault(
                "ae.decoder.final_norm.norm.num_batches_tracked",
                torch.tensor(0, dtype=torch.long),
            )
        model.load_state_dict(sd)
        model.eval().to(device)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--out", default="checkpoints/supertonic3.pt")
    args = parser.parse_args()

    ckpt = convert_all(args.onnx_dir)
    for name, sd in ckpt.items():
        n_params = sum(v.numel() for v in sd.values())
        print(f"{name}: {len(sd)} tensors, {n_params/1e6:.1f}M params")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    main()
