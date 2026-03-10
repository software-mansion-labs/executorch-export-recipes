# Whisper ExecuTorch Export 🚀

This repository provides tools to export OpenAI's **Whisper** model to the **ExecuTorch** format (`.pte`), enabling efficient on-device inference. It includes a customized model architecture optimized for mobile/edge deployment, quantization support via `torchao`, and a comprehensive testing suite.

---

## ✨ Key Model Enhancements

Our implementation is based on the [openai/whisper](https://github.com/openai/whisper) package but includes critical modifications for ExecuTorch compatibility and performance:

-   **KV Caching:** Added a persistent Key-Value cache in `MultiHeadAttention` to significantly speed up autoregressive decoding.
-   **Static Shapes & Padding:** Modified the preprocessor and encoder to handle fixed-length audio (up to 30s) to satisfy ExecuTorch's current deployment constraints.
-   **Kernel Compatibility:** Fixed non-delegated operators (like specific `LayerNorm` and `Conv1d` behaviors) to ensure the model can be correctly partitioned and lowered to backends like **XNNPACK**.
-   **SDPA Management:** Explicitly managed Scaled Dot Product Attention (SDPA) to work within the limitations of the ExecuTorch runtime.

---

## 🛠️ Usage Guide

### 1. Data Preparation
Before testing, download sample audio-transcription pairs from the LibriSpeech dataset.

```bash
python scripts/download_test_data.py -n 10 -o ./test_data
```
-   `-n, --num_samples`: Number of samples to download (default: 10).
-   `-o, --output_dir`: Directory to save `.wav` and `.json` files.

### 2. Exporting the Model
Export the model into a bundled `.pte` file containing both `encode` and `decode` methods.

```bash
python scripts/export_bundled.py --model-type tiny.en --quantize --output whisper_tiny_q.pte
```
-   `--model-type`: Type of Whisper model (e.g., `tiny.en`, `base.en`).
-   `--quantize`: Enable `int4` weight-only quantization via `torchao`.
-   `--quantize-embeddings`: (Optional) Quantize embedding layers to further reduce size.
-   `--output`: Path to the resulting `.pte` file.

### 3. Single-File Inference Test
Test the exported model on a single local file or a remote URL.

```bash
python scripts/single_test.py --model whisper_tiny_q.pte --tokenizer tokenizer.json --input ./test_data/sample_0.wav
```
-   `--model`: Path to your `.pte` file.
-   `--tokenizer`: Path to `tokenizer.json`.
-   `--input`: Local path or HTTP URL to an audio file.

### 4. Batch Evaluation (Multi-Test)
Run a full evaluation on the downloaded test dataset to see performance across multiple samples.

```bash
python scripts/multi_test.py --model whisper_tiny_q.pte --tokenizer tokenizer.json --data_dir ./test_data
```
-   `--data_dir`: Path to the directory containing your `sample_i.json` and `sample_i.wav` files.

---

## 📊 Evaluation Metrics

The `multi_test.py` script provides a detailed summary of model performance:

| Metric | Description |
| :--- | :--- |
| **WER (Word Error Rate)** | The primary accuracy metric. It measures the difference between the model's output and the reference transcription (lower is better). |
| **Encoder Time** | Time taken to process the log-mel spectrogram and generate latent features. |
| **Decoder Time/Step** | Average time for generating a single token. Indicates real-time factor for text generation. |
| **Normalization** | We use a basic normalizer (lowercasing and punctuation removal) to ensure the WER reflects semantic accuracy rather than formatting issues. |

### What is a "Good" WER?
-   **< 5%:** State-of-the-Art.
-   **5% - 15%:** Excellent for quantized mobile models (like Whisper Tiny/Base).
-   **> 25%:** May indicate issues with quantization noise or difficult audio.

---

## 🏗️ Technical Dependencies

-   **ExecuTorch:** The core runtime for running `.pte` models.
-   **torchao:** Used for advanced quantization techniques.
-   **torchaudio:** For audio loading and resampling to 16kHz.
-   **tokenizers:** Fast Rust-based tokenization.

---
*Note: This repository is a research/utility tool for exporting Whisper to edge devices efficiently.*
