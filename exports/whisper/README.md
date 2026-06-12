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
