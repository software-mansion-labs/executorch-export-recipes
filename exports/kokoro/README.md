# Kokoro Export

This directory contains basic scripts to export the Kokoro TTS model components (Duration Predictor, Synthesizer and other legacy submodules) to ExecuTorch `.pte` format.
Full content (including some extra scripts) is available within [this](https://github.com/NorbertKlockiewicz/kokoro-export) repository.

## Export Scripts

The export process is split into two parts corresponding to the two main components of the Kokoro architecture.

### 1. Duration Predictor (`export_duration_predictor.py`)

Exports the model responsible for predicting phoneme durations.

**Usage:**

```bash
python export_duration_predictor.py --input-size inp-128 --dynamic true --max-tokens 128
```

**Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `--input-size` | string | partial | One of `inp-32`, `inp-64`, `inp-128`. Required if `--bundled` is false. |
| `--bundled` | bool | No | If `true`, exports multiple entry points (methods) for different sizes into a single file. Default: `false`. |
| `--dynamic` | bool | No | Enables dynamic shapes support (experimental). Default: `false`. |
| `--max-tokens` | int | partial | Maximum number of input tokens. Tensors are being padded to this value to omit the LSTM export issue. Required if `--dynamic` is `true`. |

### 2. Synthesizer (`export_synthesizer.py`)

Exports the model responsible for generating audio waveforms from aligned text and style vectors.

**Usage:**

```bash
python export_synthesizer.py --input-size inp-128 --dynamic true --max-tokens 128 --max-duration 296
```

**Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `--input-size` | string | partial | One of `inp-32`, `inp-64`, `inp-128`. Required if `--bundled` is false. |
| `--bundled` | bool | No | If `true`, exports multiple entry points. Default: `false`. |
| `--dynamic` | bool | No | Enables dynamic shapes support. Default: `false`. |
| `--max-tokens` | int | partial | Maximum number of input tokens. Tensors are being padded to this value to omit the LSTM export issue. Required if `--dynamic` is `true`. |
| `--max-duration` | int | partial | Maximum audio duration frames. Tensors are being padded to this value to omit the LSTM export issue. Required if `--dynamic` is `true`. |

---

### 3. Legacy modules (`export_f0n_predictor.py`, `export_encoder.py`, `export_decoder.py`)

Models exported with those scripts are no longer part of the Text to Speech module in RNE - since they are already bundled into the Synthesizer model. 


## Testing (`inference_example.py`)

The `inference_example.py` script demonstrates how to run the full pipeline using `executorch`. It handles the necessary glue logic between the two models.

### Pipeline Overview

1.  **Pre-processing**
    *   **Phonemization:** Converts raw text into phonemes using `misaki`.
    *   **Tokenization:** Maps phonemes to integer IDs using a specific `MODEL_VOCAB`.
    *   **Padding:** Inputs are padded or truncated to match the exported model's input size (e.g., 128 tokens).
    *   **Voice Vector:** The reference voice style is split into `ref` (reference) and `style` vectors.

2.  **Duration Prediction**
    *   Runs the `duration_predictor` model.
    *   **Scaling:** If total predicted duration exceeds the limit (`MAX_DURATION`), the duration vector is scaled down specifically using `model.temporal_scaling.scale`.
    *   **Alignment:** Converts durations into an index map (`indices`) which aligns text tokens to audio frames.

3.  **Synthesis**
    *   Runs the `synthesizer` model using the aligned indices and style vectors.
    *   Result is a raw audio waveform.

4.  **Post-processing**
    *   **Trimming:** Uses `model.postprocessing.find_voice_bound` to detect and remove silence from the beginning and end of the generated audio.

### Running the Example

Ensure you have the `.pte` files generated or allow the script to download defaults from HuggingFace.

```bash
python inference_example.py
```
