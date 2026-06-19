#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Export the RF-DETR keypoint preview model (COCO-17 pose) to ExecuTorch .pte:
#   - CoreML fp16 / GPU : FASTEST on Apple Silicon (~42ms/inference); the demo default.
#   - CoreML fp32 / GPU : bit-exact fallback (~47ms).
#   - XNNPACK           : portable CPU (ARM/x86; ~7x slower — portability only).
#   - MLX               : Apple GPU (opt-in: MLX=1; needs executorch built from source
#                         with the MLX backend — see the README).
#
# Benchmarked on M4 Pro (2026-06) — the model is GPU-bound (its deformable attention is
# slow on the ANE; CoreML "all" already runs it on the GPU). Two findings:
#  1. fp16 on the GPU is ~1.1x faster than fp32 at sub-pixel accuracy (<0.3px) — the win.
#  2. Weight quantization does NOT help: int8/int4 only add runtime dequant, so they are
#     *slower* than fp32 (int4=49ms, int8=51ms) AND wreck keypoint accuracy. int4 is only
#     worth it for download size (19MB vs 141MB), not speed.
# Also: CoreML --min-deployment-target iOS17/iOS18 BREAKS the keypoint head — keep the
# "default" target. See the README for the full write-up.
#
# The exporter is export.py (next to this script); export_patches.py is bundled too.
# This script just runs export.py once per backend with the chosen settings, using a
# Python env that has `rfdetr` + `executorch` + `coremltools` + `flatc` installed (the
# rf-detr-keypoints repo's .venv is the easy one). Outputs land in ./exports.
#
# Usage:    ./export_keypoint_ptes.sh
# Options:  PYTHON=/path/to/venv/bin/python    # env with rfdetr+executorch+coremltools
#           REPO=/path/to/rf-detr-keypoints    # shortcut: use that checkout's .venv
#           MLX=1                              # also export the MLX backend (needs source build)
#           OUT=/somewhere                     # output dir (default: ./exports)
#           VARIANT=... NAME=...               # advanced overrides
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# The exporter (export.py) and its patches (export_patches.py) live next to this script.
# REPO is only used to locate a virtualenv that has rfdetr + executorch + coremltools
# installed; point PYTHON at any such env to avoid needing REPO at all.
EXPORT="$SCRIPT_DIR/export.py"
REPO="${REPO:-/Users/jakubchmura/Desktop/SWM_AI/rf-detr-keypoints}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
VARIANT="${VARIANT:-RFDETRKeypointPreview}"
OUT="${OUT:-$SCRIPT_DIR/exports}"
NAME="${NAME:-rfdetr_keypoint_preview}"   # files become ${NAME}_<backend>_<precision>.pte
MLX="${MLX:-0}"                            # 1 = also export MLX (requires source-built executorch)

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found/executable at $PYTHON" >&2
    echo "       Set PYTHON=/path/to/python (an env with rfdetr+executorch+coremltools)," >&2
    echo "       or REPO=/path/to/rf-detr-keypoints to use its .venv." >&2
    exit 1
fi

# The XNNPACK serializer shells out to `flatc` (FlatBuffers compiler), which lives in
# the venv's bin dir. Put it on PATH so a bare python invocation (venv not "activated")
# can find it.
VENV_BIN="$(cd "$(dirname "$PYTHON")" 2>/dev/null && pwd || true)"
[ -n "$VENV_BIN" ] && [ -x "$VENV_BIN/flatc" ] && export PATH="$VENV_BIN:$PATH"

mkdir -p "$OUT"
echo "REPO=$REPO"
echo "OUT=$OUT"
echo ""

echo "==> [1/3] CoreML fp16 / GPU  (fastest; demo default)"
"$PYTHON" "$EXPORT" --variant "$VARIANT" --backend coreml \
    --compute-unit cpu_and_gpu --compute-precision fp16 \
    --output-dir "$OUT" --output-name "${NAME}_coreml_fp16_cpu_and_gpu"

echo "==> [2/3] CoreML fp32 / GPU  (bit-exact fallback)"
"$PYTHON" "$EXPORT" --variant "$VARIANT" --backend coreml \
    --compute-unit all --compute-precision fp32 \
    --output-dir "$OUT" --output-name "${NAME}_coreml_fp32"

echo "==> [3/3] XNNPACK (portable CPU)"
"$PYTHON" "$EXPORT" --variant "$VARIANT" --backend xnnpack \
    --output-dir "$OUT" --output-name "${NAME}_xnnpack_fp32"

if [ "$MLX" = "1" ]; then
    echo "==> [bonus] MLX (Apple GPU, fp32; source-built executorch required)"
    "$PYTHON" "$EXPORT" --variant "$VARIANT" --backend mlx \
        --output-dir "$OUT" --output-name "${NAME}_mlx_fp32"
fi

echo ""
echo "==> Done. Exported .pte files:"
ls -lh "$OUT"/${NAME}_*.pte
