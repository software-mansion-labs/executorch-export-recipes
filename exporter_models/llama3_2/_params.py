"""Llama 3.2 source checkpoints, per variant.

Unlike the other LLMs here, Llama needs no vendored params and no weight
conversion: Meta publishes the native-format `params.json` and
`consolidated.00.pth` in the repo's `original/` subdirectory, and export_llama
consumes both directly. So this file is a pointer table, not a params table.

Both repos are **gated** -- accept the license on the model page and be logged
in (`hf auth login`) before exporting.

The two published variants come from two different checkpoints:

- `bf16` -- the plain Instruct weights, exported at bf16 with no PTQ.
- `spinquant` -- Meta's pre-quantized SpinQuant INT4 EO8 checkpoint. The
  weights arrive already quantized, so the export only tells executorch how to
  *load* them (`SPINQUANT` below, matching
  examples/models/llama/config/llama_xnnpack_spinquant.yaml); there is no
  qmode/group_size sweep to apply.

The quantization sweep the old `exports/llms` directory ran also measured a
plain 8da4w recipe for Llama (8da4w_gs32_emb8_64 for 1B, 8da4w_gs32_emb8_32 for
3B), but SpinQuant is what we publish as the quantized variant, so those numbers
have no variant to attach to here.
"""

MODEL_CLASS = "llama3_2"

# Loading knobs for the pre-quantized SpinQuant checkpoints.
SPINQUANT = {
    "preq_mode": "8da4w_output_8da8w",
    "preq_group_size": 32,
    "preq_embedding_quantize": "8,0",
}

SIZES = {
    "1b": {
        "bf16": "meta-llama/Llama-3.2-1B-Instruct",
        "spinquant": "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8",
    },
    "3b": {
        "bf16": "meta-llama/Llama-3.2-3B-Instruct",
        "spinquant": "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8",
    },
}
