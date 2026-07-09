#!/bin/bash
# Re-export Whisper for XNNPACK (fp32), MLX (bf16), and CoreML (fp16) with the
# dynamic-dims companion methods bundled.
# XNNPACK/CoreML use the executorch 1.2.0 .venv; MLX uses the executorch 1.3.1 .venv-mlx.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

MODELS=(tiny tiny.en base base.en small small.en)

slug() { echo "${1//./_}"; }

for m in "${MODELS[@]}"; do
  s=$(slug "$m")
  echo "========== XNNPACK fp32: $m =========="
  .venv/bin/python export_xnnpack.py --model-type "$m" --dtype float32 \
    --output "output/whisper_${s}_xnnpack_fp32.pte" 2>&1 | tail -3
done

for m in "${MODELS[@]}"; do
  s=$(slug "$m")
  echo "========== CoreML fp16: $m =========="
  .venv/bin/python export_coreml.py --model-type "$m" --dtype float16 \
    --output "output/whisper_${s}_coreml_fp16.pte" 2>&1 | tail -3
done

for m in "${MODELS[@]}"; do
  s=$(slug "$m")
  echo "========== MLX bf16: $m =========="
  .venv-mlx/bin/python export_mlx.py --model-type "$m" --dtype bfloat16 \
    --output "output/whisper_${s}_mlx_bf16.pte" 2>&1 \
    | grep -E "DELEGATION|Success|Error|Traceback|Export completed" | tail -6
done

echo "========== DONE =========="
ls -la output/
