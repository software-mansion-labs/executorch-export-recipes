# llama3_2

## Gated weights

Both the Instruct and the SpinQuant repos are gated on HuggingFace. Accept the
license on each model page and `hf auth login` before exporting, or
`snapshot_download` fails with a 401/403.

## No weight conversion

Meta ships native-format `params.json` + `consolidated.00.pth` (under
`original/` for the Instruct repos, at the root for the SpinQuant ones), which
export_llama reads directly. That is why this model has no `_params.py` params
table and no HF->Meta converter -- unlike every other LLM here.

## What is not exported

- **QLoRA.** The published repo still carries `llama-3.2-{1B,3B}/QLoRA/*.pte`
  from the v0.8.0 layout, and react-native-executorch still pins those URLs at
  the `v0.8.0` tag. They are superseded by SpinQuant as the quantized variant
  and are absent from the current `{size}/xnnpack/config.json`, so there is no
  variant here to export them into. The recipe was
  `llama_xnnpack_qat.yaml` (`use_qat: True`, `use_lora: 16`) against the
  `*-QLORA_INT4_EO8` checkpoints if it ever needs restoring.
- **Plain 8da4w.** Measured by the old quantization sweep but never published
  for Llama -- see `_params.py`.

## Published-config discrepancy: get_max_context_len

The published config declares `get_max_seq_len: 2048` but
`get_max_context_len: 4096`, while the old `exports/llms` recipe exported with
`--max-seq-length 2048 --max-context-length 2048`. `export.py` takes both from
the config (runner constants are the config's job), so a re-export produces a
4096-token KV cache -- larger than the shipped `.pte` was built with.

Unlike the `phi_4_mini` case, this is not self-contradictory (a context longer
than the max prefill chunk is normal for a KV-cache runner), so the config is
left fetched rather than vendored. If the intent was 2048, fix it in the
published config rather than here.
