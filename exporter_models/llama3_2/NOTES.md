# llama3_2

## Gated weights

Both the Instruct and the SpinQuant repos are gated on HuggingFace. Accept the
license on each model page and `hf auth login` before exporting, or
`snapshot_download` fails with a 401/403.

## No weight conversion

Meta ships native-format `params.json` + `consolidated.00.pth` (under
`original/` for the Instruct repos, at the root for the SpinQuant ones), which
export_llama reads directly. That is why this model has no params table in
`_params.py` and no HF->Meta converter, unlike every other LLM here.

## Config is vendored, not fetched

The two repo-wide errors, corrected: `get_n_layers: null` where the `.pte`
carries 16 / 28, and a `forward` sequence dim of 2048 where the `.pte` declares
2047.

## Published artifacts: the two variants have different context budgets

| variant | .pte `get_max_context_len` |
|---|---|
| `llama_3_2_{1b,3b}_xnnpack_bf16.pte` | `4096` |
| `llama_3_2_{1b,3b}_xnnpack_spinquant.pte` | `2048` |

The published config declares `4096`, so it describes the bf16 files and
over-promises the spinquant ones -- a runner trusting it would size a 4096-token
KV budget against a `.pte` built for 2048. (The old `exports/llms` recipe
exported everything at `--max-seq-length 2048 --max-context-length 2048`, so the
bf16 files are the ones that drifted, upward.)

This config keeps `4096` for both rather than cutting bf16 in half: the longer
context works and is what users have today. The two spinquant files need a
re-export to match. Everything else matches.

Check with `tools/probe_published_pte.py llama3_2 --size 1b`.

## What is not exported

- **QLoRA.** The published repo still carries `llama-3.2-{1B,3B}/QLoRA/*.pte`
  from the v0.8.0 layout, and react-native-executorch still pins those URLs at
  the `v0.8.0` tag. They are superseded by SpinQuant as the quantized variant
  and are absent from the current `{size}/xnnpack/config.json`, so there is no
  variant here to export them into. The recipe was `llama_xnnpack_qat.yaml`
  (`use_qat: True`, `use_lora: 16`) against the `*-QLORA_INT4_EO8` checkpoints
  if it ever needs restoring.
- **Plain 8da4w.** Measured by the old sweep (`8da4w_gs32_emb8_64` for 1B,
  `8da4w_gs32_emb8_32` for 3B) but never published for Llama, so there is no
  variant to attach it to.
