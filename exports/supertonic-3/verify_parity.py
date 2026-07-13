"""Numerical parity check: PyTorch port vs ONNX Runtime.

Usage: python -m supertonic_torch.verify_parity [--onnx-dir onnx]
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from .convert import load_models
from .pipeline import UnicodeProcessor, Style, length_to_mask


def report(name, ref, out, tol=2e-3):
    ref, out = np.asarray(ref), np.asarray(out)
    adiff = np.abs(ref - out)
    denom = np.abs(ref).max() + 1e-9
    rel = adiff.max() / denom
    status = "OK " if rel < tol else "FAIL"
    print(f"[{status}] {name:22s} shape={list(ref.shape)} max_abs_diff={adiff.max():.3e} rel={rel:.3e}")
    return rel < tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--styles-dir", default="voice_styles")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    proc = UnicodeProcessor(f"{args.onnx_dir}/unicode_indexer.json")
    style = Style.from_json_file(f"{args.styles_dir}/M1.json")

    texts = ["Hello world, this is a parity test of the PyTorch port!",
             "Short one."]
    text_ids, text_mask = proc(texts, lang="en")
    style_ttl = np.repeat(style.ttl, 2, axis=0)
    style_dp = np.repeat(style.dp, 2, axis=0)

    models = load_models(onnx_dir=args.onnx_dir)
    sess = lambda n: ort.InferenceSession(
        f"{args.onnx_dir}/{n}.onnx", providers=["CPUExecutionProvider"]
    )
    ok = True

    # --- duration predictor -------------------------------------------------
    dp_ref, *_ = sess("duration_predictor").run(
        None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask}
    )
    with torch.inference_mode():
        dp_out = models["duration_predictor"](
            torch.from_numpy(text_ids),
            torch.from_numpy(style_dp),
            torch.from_numpy(text_mask),
        ).numpy()
    ok &= report("duration_predictor", dp_ref, dp_out)

    # --- text encoder --------------------------------------------------------
    te_ref, *_ = sess("text_encoder").run(
        None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask}
    )
    with torch.inference_mode():
        te_out = models["text_encoder"](
            torch.from_numpy(text_ids),
            torch.from_numpy(style_ttl),
            torch.from_numpy(text_mask),
        ).numpy()
    ok &= report("text_encoder", te_ref, te_out)

    # --- vector estimator (uses ONNX text_emb as common input) ---------------
    bsz = len(texts)
    latent_len = 40
    latent_dim = 144
    noisy = rng.standard_normal((bsz, latent_dim, latent_len)).astype(np.float32)
    latent_lengths = np.array([latent_len, latent_len - 13], dtype=np.int64)
    latent_mask = length_to_mask(latent_lengths, latent_len)
    noisy = noisy * latent_mask
    current = np.array([2.0, 2.0], dtype=np.float32)
    total = np.array([8.0, 8.0], dtype=np.float32)

    ve_ref, *_ = sess("vector_estimator").run(
        None,
        {
            "noisy_latent": noisy, "text_emb": te_ref, "style_ttl": style_ttl,
            "text_mask": text_mask, "latent_mask": latent_mask,
            "current_step": current, "total_step": total,
        },
    )
    with torch.inference_mode():
        ve_out = models["vector_estimator"](
            torch.from_numpy(noisy), torch.from_numpy(te_ref),
            torch.from_numpy(style_ttl), torch.from_numpy(text_mask),
            torch.from_numpy(latent_mask), torch.from_numpy(current),
            torch.from_numpy(total),
        ).numpy()
    ok &= report("vector_estimator", ve_ref, ve_out)

    # --- vocoder --------------------------------------------------------------
    latent = (rng.standard_normal((1, latent_dim, 24)) * 0.25).astype(np.float32)
    vo_ref, *_ = sess("vocoder").run(None, {"latent": latent})
    with torch.inference_mode():
        vo_out = models["vocoder"](torch.from_numpy(latent)).numpy()
    ok &= report("vocoder", vo_ref, vo_out, tol=5e-3)

    # --- end-to-end (shared noise) --------------------------------------------
    dur = dp_ref / 1.05
    sr, chunk = 44100, 512 * 6
    wav_lengths = (dur * sr).astype(np.int64)
    L = int((dur.max() * sr + chunk - 1) // chunk)
    xt0 = (rng.standard_normal((bsz, latent_dim, L))).astype(np.float32)
    lmask = length_to_mask((wav_lengths + chunk - 1) // chunk, L)
    xt_ref = xt0 * lmask
    xt_out = torch.from_numpy(xt_ref.copy())

    ve_sess = sess("vector_estimator")
    total = np.array([8.0] * bsz, dtype=np.float32)
    with torch.inference_mode():
        for step in range(8):
            cur = np.array([float(step)] * bsz, dtype=np.float32)
            xt_ref, *_ = ve_sess.run(
                None,
                {
                    "noisy_latent": xt_ref, "text_emb": te_ref,
                    "style_ttl": style_ttl, "text_mask": text_mask,
                    "latent_mask": lmask, "current_step": cur, "total_step": total,
                },
            )
            xt_out = models["vector_estimator"](
                xt_out, torch.from_numpy(te_out), torch.from_numpy(style_ttl),
                torch.from_numpy(text_mask), torch.from_numpy(lmask),
                torch.from_numpy(cur), torch.from_numpy(total),
            )
        wav_ref, *_ = sess("vocoder").run(None, {"latent": xt_ref})
        wav_out = models["vocoder"](xt_out).numpy()
    ok &= report("end2end latent (8 steps)", xt_ref, xt_out.numpy(), tol=5e-3)
    ok &= report("end2end waveform", wav_ref, wav_out, tol=2e-2)

    print("\nPARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
