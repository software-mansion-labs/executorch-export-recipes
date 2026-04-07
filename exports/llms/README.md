# LLM Export

This directory serves two main purposes. First, it provides reproducible
environment and recipes to export a variety of LLMs that we provide in
`react-native-executorch`. These recipes include all necessary steps to convert
HuggingFace checkpoints into a format compatible with executorch, along with the
appropriate configuration files and metadata. Second, it includes a benchmarking
script `llm_export_benchmark.py` that can be used to export and benchmark
different quantization schemes to make informed decisions about which to use for
a given model.

## How to Use

### Prerequisites

**ExecuTorch** must be installed from source inside its own virtualenv:

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
python3.12 -m venv .venv
source .venv/bin/activate
./install_executorch.sh   
```

The script expects a `.venv` at `<executorch-dir>/.venv` and will invoke
`<executorch-dir>/.venv/bin/python` directly to run exports.

### Setup

Install dependencies using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Export recipes reference

All commands assume you run them from this directory. Model checkpoints are
downloaded to `./models/`.

> **HuggingFace access:** Some models require HuggingFace access approval. If
> you have access to a model but encounter permission errors with `hf download`,
> you may need to set up HuggingFace CLI authentication.
---
> **ExecuTorch directory:** Set the environment variable `EXECUTORCH_DIR` to the
> path of your local executorch clone before running export commands.
---
> **Extra param files required:** The `jsons/` directory contains custom
> `*_config.json` files for model sizes not bundled with executorch
> (`smollm2/360M`, `smollm2/1.7B`, `qwen2_5/3B`, `qwen2_5/7B`). Copy them to
> the appropriate locations before exporting:
>
> ```bash
> cp jsons/smollm2/*.json "$EXECUTORCH_DIR/examples/models/smollm2/"
> cp jsons/qwen2_5/*.json "$EXECUTORCH_DIR/examples/models/qwen2_5/config/"
> ```
>
> `jsons/lfm2_5/` and `jsons/hammer2_1/` configs are referenced directly by
> path in the export commands below — no copy needed.
>
---

### [`LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M)

`LFM2.5-350M` is a general-purpose instruction-tuned model (no separate Instruct
variant — the base model is published separately as
[`LFM2.5-350M-Base`](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)).

```bash
hf download LiquidAI/LFM2.5-350M \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/LFM2.5-350M
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/lfm2/convert_weights.py" \
  ./models/LFM2.5-350M ./models/LFM2.5-350M.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/lfm2_5_xnnpack.yaml \
  --model-class lfm2_5_1_2b \
  --model-params ./jsons/lfm2_5/lfm2_5_350m_config.json \
  --checkpoint ./models/LFM2.5-350M.pth \
  --metadata '{"get_bos_id":1,"get_eos_ids":[7]}' \
  --output-dir exports/lfm2_5_350m \
  --tokenizer LiquidAI/LFM2.5-350M \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)

```bash
hf download LiquidAI/LFM2.5-1.2B-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/LFM2.5-1.2B-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/lfm2/convert_weights.py" \
  ./models/LFM2.5-1.2B-Instruct ./models/LFM2.5-1.2B-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/lfm2_5_xnnpack.yaml \
  --model-class lfm2_5_1_2b \
  --model-params "$EXECUTORCH_DIR/examples/models/lfm2/config/lfm2_5_1_2b_config.json" \
  --checkpoint ./models/LFM2.5-1.2B-Instruct.pth \
  --metadata '{"get_bos_id":1,"get_eos_ids":[7]}' \
  --output-dir exports/lfm2_5_1_2b \
  --tokenizer LiquidAI/LFM2.5-1.2B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen2.5-0.5B-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Qwen2.5-0.5B-Instruct ./models/Qwen2.5-0.5B-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_0_5b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen2_5/config/0_5b_config.json" \
  --checkpoint ./models/Qwen2.5-0.5B-Instruct.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151645]}' \
  --output-dir exports/qwen2_5_0_5b \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen2.5-1.5B-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Qwen2.5-1.5B-Instruct ./models/Qwen2.5-1.5B-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_1_5b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen2_5/config/1_5b_config.json" \
  --checkpoint ./models/Qwen2.5-1.5B-Instruct.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151645]}' \
  --output-dir exports/qwen2_5_1_5b \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

```bash
hf download Qwen/Qwen2.5-3B-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen2.5-3B-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Qwen2.5-3B-Instruct ./models/Qwen2.5-3B-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_1_5b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen2_5/config/3b_config.json" \
  --checkpoint ./models/Qwen2.5-3B-Instruct.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151643]}' \
  --output-dir exports/qwen2_5_3b \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Hammer2.1-0.5b`](https://huggingface.co/MadeAgents/Hammer2.1-0.5b)

Hammer2.1 is a fine-tune of Qwen2.5 optimised for function calling. It uses the
same architecture as Qwen2.5.

```bash
hf download MadeAgents/Hammer2.1-0.5b \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Hammer2.1-0.5b
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Hammer2.1-0.5b ./models/Hammer2.1-0.5b.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_0_5b \
  --model-params ./jsons/hammer2_1/hammer2_1_0_5b_config.json \
  --checkpoint ./models/Hammer2.1-0.5b.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151645]}' \
  --output-dir exports/hammer2_1_0_5b \
  --tokenizer MadeAgents/Hammer2.1-0.5b \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Hammer2.1-1.5b`](https://huggingface.co/MadeAgents/Hammer2.1-1.5b)

```bash
hf download MadeAgents/Hammer2.1-1.5b \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Hammer2.1-1.5b
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Hammer2.1-1.5b ./models/Hammer2.1-1.5b.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_1_5b \
  --model-params ./jsons/hammer2_1/hammer2_1_1_5b_config.json \
  --checkpoint ./models/Hammer2.1-1.5b.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151645]}' \
  --output-dir exports/hammer2_1_1_5b \
  --tokenizer MadeAgents/Hammer2.1-1.5b \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Hammer2.1-3b`](https://huggingface.co/MadeAgents/Hammer2.1-3b)

```bash
hf download MadeAgents/Hammer2.1-3b \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Hammer2.1-3b
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen2_5/convert_weights.py" \
  ./models/Hammer2.1-3b ./models/Hammer2.1-3b.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen2_5_xnnpack.yaml \
  --model-class qwen2_5_1_5b \
  --model-params ./jsons/hammer2_1/hammer2_1_3b_config.json \
  --checkpoint ./models/Hammer2.1-3b.pth \
  --metadata '{"get_bos_id":151643,"get_eos_ids":[151645]}' \
  --output-dir exports/hammer2_1_3b \
  --tokenizer MadeAgents/Hammer2.1-3b \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)

Qwen3 models are unified (no separate Instruct variant) — base models support both
thinking and non-thinking modes.

```bash
hf download Qwen/Qwen3-0.6B \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen3-0.6B
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen3/convert_weights.py" \
  ./models/Qwen3-0.6B ./models/Qwen3-0.6B.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen3_xnnpack.yaml \
  --model-class qwen3_0_6b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen3/config/0_6b_config.json" \
  --checkpoint ./models/Qwen3-0.6B.pth \
  --metadata '{"get_bos_id":151644,"get_eos_ids":[151645]}' \
  --output-dir exports/qwen3_0_6b \
  --tokenizer Qwen/Qwen3-0.6B \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)

Qwen3 models are unified (no separate Instruct variant) — base models support both
thinking and non-thinking modes.

```bash
hf download Qwen/Qwen3-1.7B \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen3-1.7B
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen3/convert_weights.py" \
  ./models/Qwen3-1.7B ./models/Qwen3-1.7B.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen3_xnnpack.yaml \
  --model-class qwen3_1_7b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen3/config/1_7b_config.json" \
  --checkpoint ./models/Qwen3-1.7B.pth \
  --metadata '{"get_bos_id":151644,"get_eos_ids":[151645]}' \
  --output-dir exports/qwen3_1_7b \
  --tokenizer Qwen/Qwen3-1.7B \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)

Qwen3 models are unified (no separate Instruct variant) — base models support both
thinking and non-thinking modes.

```bash
hf download Qwen/Qwen3-4B \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --local-dir ./models/Qwen3-4B
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/qwen3/convert_weights.py" \
  ./models/Qwen3-4B ./models/Qwen3-4B.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/qwen3_xnnpack.yaml \
  --model-class qwen3_4b \
  --model-params "$EXECUTORCH_DIR/examples/models/qwen3/config/4b_config.json" \
  --checkpoint ./models/Qwen3-4B.pth \
  --metadata '{"get_bos_id":151644,"get_eos_ids":[151645]}' \
  --output-dir exports/qwen3_4b \
  --tokenizer Qwen/Qwen3-4B \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

The `original/` subdirectory contains native-format `params.json` and `consolidated.00.pth`
— no conversion needed (requires HuggingFace access approval):

```bash
hf download meta-llama/Llama-3.2-1B-Instruct \
  --include "original/*" \
  --local-dir ./models/Llama-3.2-1B-Instruct
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/llama3_2_xnnpack.yaml \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-1B-Instruct/original/params.json \
  --checkpoint ./models/Llama-3.2-1B-Instruct/original/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_1b \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

The `original/` subdirectory contains native-format `params.json` and `consolidated.00.pth`
— no conversion needed (requires HuggingFace access approval):

```bash
hf download meta-llama/Llama-3.2-3B-Instruct \
  --include "original/*" \
  --local-dir ./models/Llama-3.2-3B-Instruct
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/llama3_2_xnnpack.yaml \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-3B-Instruct/original/params.json \
  --checkpoint ./models/Llama-3.2-3B-Instruct/original/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_3b \
  --tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8)

Pre-quantized SpinQuant INT4 checkpoint. Uses a dedicated executorch config that sets
`use_spin_quant: native` and `preq_mode: preq_8da4w_out_8da8w` — no extra quantization
overrides needed.

```bash
hf download meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8 \
  --include "consolidated.00.pth" \
  --include "params.json" \
  --local-dir ./models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml "$EXECUTORCH_DIR/examples/models/llama/config/llama_xnnpack_spinquant.yaml" \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8/params.json \
  --checkpoint ./models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_1b_spinquant \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8)

Pre-quantized SpinQuant INT4 checkpoint. Uses a dedicated executorch config that sets
`use_spin_quant: native` and `preq_mode: preq_8da4w_out_8da8w` — no extra quantization
overrides needed.

```bash
hf download meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8 \
  --include "consolidated.00.pth" \
  --include "params.json" \
  --local-dir ./models/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml "$EXECUTORCH_DIR/examples/models/llama/config/llama_xnnpack_spinquant.yaml" \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8/params.json \
  --checkpoint ./models/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_3b_spinquant \
  --tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-1B-Instruct-QLORA_INT4_EO8`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8)

Pre-quantized QAT+LoRA INT4 checkpoint. Uses a dedicated executorch config that sets
`use_qat: True` and `use_lora: 16` — no extra quantization overrides needed.

```bash
hf download meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8 \
  --include "consolidated.00.pth" \
  --include "params.json" \
  --local-dir ./models/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml "$EXECUTORCH_DIR/examples/models/llama/config/llama_xnnpack_qat.yaml" \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-1B-Instruct-QLORA_INT4_EO8/params.json \
  --checkpoint ./models/Llama-3.2-1B-Instruct-QLORA_INT4_EO8/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_1b_qat \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Llama-3.2-3B-Instruct-QLORA_INT4_EO8`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8)

Pre-quantized QAT+LoRA INT4 checkpoint. Uses a dedicated executorch config that sets
`use_qat: True` and `use_lora: 16` — no extra quantization overrides needed.

```bash
hf download meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8 \
  --include "consolidated.00.pth" \
  --include "params.json" \
  --local-dir ./models/Llama-3.2-3B-Instruct-QLORA_INT4_EO8
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml "$EXECUTORCH_DIR/examples/models/llama/config/llama_xnnpack_qat.yaml" \
  --model-class llama3_2 \
  --model-params ./models/Llama-3.2-3B-Instruct-QLORA_INT4_EO8/params.json \
  --checkpoint ./models/Llama-3.2-3B-Instruct-QLORA_INT4_EO8/consolidated.00.pth \
  --metadata '{"get_bos_id":128000,"get_eos_ids":[128009,128001]}' \
  --output-dir exports/llama3_2_3b_qat \
  --tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct)

```bash
hf download microsoft/Phi-4-mini-instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --include "config.json" \
  --local-dir ./models/Phi-4-mini-instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/phi_4_mini/convert_weights.py" \
  ./models/Phi-4-mini-instruct ./models/Phi-4-mini-instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/phi4mini_xnnpack.yaml \
  --model-class phi_4_mini \
  --model-params "$EXECUTORCH_DIR/examples/models/phi_4_mini/config/config.json" \
  --checkpoint ./models/Phi-4-mini-instruct.pth \
  --metadata '{"get_bos_id":199999,"get_eos_ids":[199999]}' \
  --output-dir exports/phi_4_mini \
  --tokenizer microsoft/Phi-4-mini-instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

```bash
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --include "config.json" \
  --local-dir ./models/SmolLM2-135M-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/smollm2/convert_weights.py" \
  ./models/SmolLM2-135M-Instruct ./models/SmolLM2-135M-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/smollm2_xnnpack.yaml \
  --model-class smollm2 \
  --model-params "$EXECUTORCH_DIR/examples/models/smollm2/135M_config.json" \
  --checkpoint ./models/SmolLM2-135M-Instruct.pth \
  --metadata '{"get_bos_id":1,"get_eos_ids":[2]}' \
  --output-dir exports/smollm2_135m \
  --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

```bash
hf download HuggingFaceTB/SmolLM2-360M-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --include "config.json" \
  --local-dir ./models/SmolLM2-360M-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/smollm2/convert_weights.py" \
  ./models/SmolLM2-360M-Instruct ./models/SmolLM2-360M-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/smollm2_xnnpack.yaml \
  --model-class smollm2 \
  --model-params "$EXECUTORCH_DIR/examples/models/smollm2/360M_config.json" \
  --checkpoint ./models/SmolLM2-360M-Instruct.pth \
  --metadata '{"get_bos_id":1,"get_eos_ids":[2]}' \
  --output-dir exports/smollm2_360m \
  --tokenizer HuggingFaceTB/SmolLM2-360M-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

### [`SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

```bash
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --include "*.safetensors" \
  --include "*.safetensors.index.json" \
  --include "config.json" \
  --local-dir ./models/SmolLM2-1.7B-Instruct
"$EXECUTORCH_DIR/.venv/bin/python" "$EXECUTORCH_DIR/examples/models/smollm2/convert_weights.py" \
  ./models/SmolLM2-1.7B-Instruct ./models/SmolLM2-1.7B-Instruct.pth
```

```bash
uv run python llm_export_benchmark.py \
  --executorch-dir "$EXECUTORCH_DIR" \
  --config-yaml ./configs/smollm2_xnnpack.yaml \
  --model-class smollm2 \
  --model-params "$EXECUTORCH_DIR/examples/models/smollm2/1_7B_config.json" \
  --checkpoint ./models/SmolLM2-1.7B-Instruct.pth \
  --metadata '{"get_bos_id":1,"get_eos_ids":[2]}' \
  --output-dir exports/smollm2_1_7b \
  --tokenizer HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --max-seq-length 2048 \
  --max-context-length 2048
```

## Additional models via Optimum ExecuTorch

Some HuggingFace models can be exported with `optimum-cli export executorch`
even when this repository does not provide dedicated `export_llm` configs
(`--config-yaml`, `--model-class`, `--model-params`).

This applies to:

- [`speakleash/Bielik-1.5B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct/tree/main)
- [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m)
- [`google/gemma-3-1b`](https://huggingface.co/google/gemma-3-1b)

> These models are gated on HuggingFace. You must accept access terms on each
> model page first, otherwise downloads/export will fail with.

### [`Bielik-1.5B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct/tree/main)

```bash
uv run optimum-cli export executorch \
  --model speakleash/Bielik-1.5B-v3.0-Instruct \
  --task text-generation \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w \
  --output_dir exports/bielik_1_5b_v3_0_instruct_optimum
```

### [`gemma-3-270m`](https://huggingface.co/google/gemma-3-270m)

```bash
uv run optimum-cli export executorch \
  --model google/gemma-3-270m \
  --task text-generation \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w \
  --output_dir exports/gemma_3_270m_optimum
```

## Results

The benchmarks were run on a machine with Intel Core i9-12900K CPU and 64GB RAM.
Note that having enough RAM (at least 32GB) is crucial for running benchmarks
for larger models. The following configurations were used for benchmarking:

```python
PROMPTS = [
    "In a distant galaxy far from our own, there existed",
    "The history of artificial intelligence began in the mid-twentieth century",
    "Once upon a time in a small village nestled between two mountains",
    "Scientists have recently discovered a new species of deep-sea creature",
    "The economic impact of climate change on developing nations",
]
BENCHMARKS: list[Benchmark] = [
    SizeBenchmark(),
    ThroughputBenchmark(prompts=PROMPTS, max_new_tokens=128),
    PerplexityBenchmark(dataset="wikitext", config="wikitext-2-raw-v1", split="test", max_tokens=1000),
    KLDivergenceBenchmark(dataset="wikitext", config="wikitext-2-raw-v1", split="test", max_tokens=1000),
]
QUANTIZATION_CONFIGS: dict[str, list[str]] = {
    "bf16": ["++model.dtype_override=bf16"],
    "8da4w_default": ["++quantization.qmode=8da4w"],
    **{
        f"8da4w_gs{lin_gs}_emb8_{emb_gs}": [
            "++quantization.qmode=8da4w",
            f"++quantization.group_size={lin_gs}",
            f"++quantization.embedding_quantize='8,{emb_gs}'",
        ]
        for emb_gs in (0, 32, 64)
        for lin_gs in (0, 32, 64)
    },
}
```

The [results](results/) directory contains detailed benchmark outputs for each
model and quantization configuration, including raw csv files and interactive
graphs with plotted Pareto frontiers.

### Best Quantization Configs Summary

The table below provides one recommended quantization setup per model based on
the benchmark outputs in `results/*/results.csv`.

| Model           | Recommended quant config |
|-----------------|--------------------------|
| hammer2_1_0_5b  | 8da4w_gs32_emb8_0        |
| hammer2_1_1_5b  | 8da4w_gs32_emb8_32       |
| hammer2_1_3b    | 8da4w_gs32_emb8_0        |
| lfm2_5_1_2b     | 8da4w_gs32_emb8_0        |
| lfm2_5_350m     | 8da4w_gs32_emb8_32       |
| llama3_2_1b     | 8da4w_gs32_emb8_64       |
| llama3_2_3b     | 8da4w_gs32_emb8_32       |
| phi_4_mini      | 8da4w_gs32_emb8_64       |
| qwen2_5_0_5b    | 8da4w_gs32_emb8_0        |
| qwen2_5_1_5b    | 8da4w_gs32_emb8_32       |
| qwen2_5_3b      | 8da4w_gs32_emb8_32       |
| qwen3_0_6b      | 8da4w_gs32_emb8_64       |
| qwen3_1_7b      | 8da4w_gs32_emb8_64       |
| qwen3_4b        | 8da4w_gs32_emb8_64       |
| smollm2_135m    | 8da4w_default            |
| smollm2_1_7b    | 8da4w_gs64_emb8_64       |
| smollm2_360m    | 8da4w_default            |
