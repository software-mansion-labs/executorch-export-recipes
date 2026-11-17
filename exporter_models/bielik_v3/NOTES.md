# Bielik v3 1.5B

Source: `speakleash/Bielik-1.5B-v3.0-Instruct` (gated; requires HF access).
Weights retain their upstream Apache-2.0 terms. The local conversion recipe
maps the checkpoint to ExecuTorch's Llama model. Context is 8192, with 2048
prefill and float32 public logits for both bf16 and 8da4w compute recipes.

BOS 1 and EOS [4, 2] follow the publisher's generation config:
https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct-FP8-Dynamic/blob/41859e61d2088a919c2a0073e4a2859f2cdac488/generation_config.json

No new artifact has been exported or checked on device during review fixes.
Upstream checkpoint access is unavailable in the review environment. Graph
parity, quantized task quality, delegation, and target-device latency remain
unverified. Do not interpret registry discovery or dry-run success as runtime
validation.
