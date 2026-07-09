# Whisper Export Scripts

Export and evaluate Whisper models using [ExecuTorch](https://github.com/pytorch/executorch).

The Whisper model used here (`custom_whisper/`) is a modified version of the
[openai/whisper](https://github.com/openai/whisper) package with a custom KV-cache
implementation, adapted for mobile/edge deployment.

## Scripts

### `download_test_data.py`

Downloads audio samples from LibriSpeech for evaluation.

```
python download_test_data.py -n 10 -o test_data
```

| Option         | Default       | Description                  |
| -------------- | ------------- | ---------------------------- |
| `-n, --num_samples` | `10`       | Number of samples to download |
| `-o, --output_dir`  | `test_data` | Output directory              |

### `export_coreml.py`

Exports a Whisper model as a bundled `.pte` file with a **CoreML** backend (Apple devices).

```
python export_coreml.py --output whisper.pte [--model-type tiny.en] [--dtype float32]
```

| Option            | Default    | Choices                              |
| ----------------- | ---------- | ------------------------------------ |
| `--model-type`    | `tiny.en`  | Any Whisper model (e.g. `turbo`)     |
| `--dtype`         | `float32`  | `float16`, `bfloat16`, `float32`     |
| `--output`        | *(required)* | Output `.pte` path                  |
| `--int8-weights`  | off        | Enable CoreML int8 per-channel quant |

### `export_xnnpack.py`

Exports a Whisper model as a bundled `.pte` file with an **XNNPACK** backend (CPU).

```
python export_xnnpack.py --output whisper.pte [--model-type tiny.en] [--dtype float32]
```

| Option                | Default      | Description                          |
| --------------------- | ------------ | ------------------------------------ |
| `--model-type`        | `tiny.en`    | Whisper model type                   |
| `--dtype`             | `float32`    | `float16`, `bfloat16`, `float32`     |
| `--quantize`          | off          | Enable torchao int4 weight-only quant|
| `--group-size`        | `64`         | Group size for quantization           |
| `--quantize-embeddings` | off        | Quantize embeddings                   |
| `--output`            | *(required)* | Output `.pte` path                   |

### `export_mlx.py`

Exports a Whisper model as a bundled `.pte` file with the **MLX** backend
(Apple Silicon GPU via [MLX](https://github.com/ml-explore/mlx)). The decoder is
rebuilt from `custom_whisper.model_mlx.TextDecoderMlx` so the KV-cache writes and
self-attention fully delegate to MLX (`mlx::kv_cache_update` + `mlx::custom_sdpa`).

```
python export_mlx.py --output whisper.pte [--model-type tiny.en] [--dtype bfloat16]
```

| Option              | Default      | Description                                   |
| ------------------- | ------------ | --------------------------------------------- |
| `--model-type`      | `tiny.en`    | Whisper model type                            |
| `--dtype`           | `float32`    | `float16`, `bfloat16`, `float32` (use `bfloat16`) |
| `--max-cache-len`   | model max    | Self-attention KV cache length                |
| `--legacy-decoder`  | off          | Use the buffer-based `TextDecoder` (not delegated) |
| `--no-encoder-sdpa` | off          | Disable SDPA on the encoder                   |
| `--output`          | *(required)* | Output `.pte` path                            |

> **Requires the MLX backend.** The MLX delegate ships only with `executorch>=1.3.1`
> (`from executorch.backends.mlx import MLXPartitioner`) on Apple Silicon — it is
> **not** in the `1.2.0` env used by the XNNPACK/CoreML scripts. Use a separate
> venv (`.venv-mlx`) with `executorch==1.3.1`.

> **Dynamic input shapes.** All three export scripts additionally bundle two
> zero-argument companion methods, `get_dynamic_dims_encode` and
> `get_dynamic_dims_decode`. Each returns, per tensor input of the corresponding
> method, a `[rank, 3]` int32 tensor of `[min, max, step]` per axis (static axes
> use `[size, size, 1]`). The `react-native-executorch` runtime reads these to
> recover the live symbolic range of dynamic dimensions (e.g. the variable audio
> length `T_audio` on `encode`), which ExecuTorch does not otherwise expose at
> runtime. See `model.cpp::parseDynamicInputShapes` in the RNE package.

### `multi_test.py`

Evaluates a bundled `.pte` model against a directory of test samples (audio + transcript JSON).
Reports Word Error Rate (WER) and throughput.

```
python multi_test.py --model whisper.pte --tokenizer tokenizer.json --data_dir test_data
```

| Option        | Description                             |
| ------------- | --------------------------------------- |
| `--model`     | Path to bundled `.pte` model            |
| `--tokenizer` | Path to `tokenizer.json`                |
| `--data_dir`  | Directory with `sample_*.wav` / `sample_*.json` |

## Typical Workflow

```bash
# 1. Download test data
python download_test_data.py -n 5 -o test_data

# 2. Export the model (choose a backend)
python export_xnnpack.py --model-type tiny.en --output whisper.pte

# 3. Run evaluation
python multi_test.py --model whisper.pte --tokenizer tokenizer.json --data_dir test_data
```
