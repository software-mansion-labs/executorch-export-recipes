#!/usr/bin/env bash
# Re-export every OCR artifact under the current get_model_schema contract.
# Each model runs inside its own locked uv env. Logs land in exports/logs/.
#   tools/export_ocr_all.sh [pp|layout|helpers|easy]...   (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=exports
LOGS=$OUT/logs
mkdir -p "$LOGS"

run() { # run <model> <backend> [size]
  local model=$1 backend=$2 size=${3:-}
  local tag="$model${size:+-$size}-$backend"
  local args=(export "$model" --backend "$backend" --output-dir "$OUT/$model${size:+/$size}")
  [ -n "$size" ] && args+=(--size "$size")
  printf '>> %-34s ' "$tag"
  if uv run --project "exporter_models/$model" exporter "${args[@]}" >"$LOGS/$tag.log" 2>&1; then
    echo "ok"
  else
    echo "FAILED (see $LOGS/$tag.log)"
  fi
}

BACKENDS=(xnnpack coreml vulkan)
LANGS=(english latin japanese zh_sim korean telugu kannada cyrillic)

for group in "${@:-pp layout helpers easy}"; do
  case $group in
    pp)      for b in "${BACKENDS[@]}"; do run paddleocr "$b"; done ;;
    layout)  for b in "${BACKENDS[@]}"; do run paddleocr_layout "$b"; done ;;
    helpers) for b in "${BACKENDS[@]}"; do run paddleocr_doc_helpers "$b"; done ;;
    easy)    for l in "${LANGS[@]}"; do for b in "${BACKENDS[@]}"; do run easyocr "$b" "$l"; done; done ;;
    *)       echo "unknown group '$group'" >&2; exit 2 ;;
  esac
done
