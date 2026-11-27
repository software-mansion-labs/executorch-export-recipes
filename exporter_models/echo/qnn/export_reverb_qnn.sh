#!/bin/bash
# The QNN programs react-native-executorch ships for Reverb (Reverb-TTS model/executorch/qnn/int4/): the int4 backbone
# with memory-planned outputs, one file per SoC, named by its Hexagon version. One export at a time: each peaks at
# ~21 GB of RAM and takes ~6 min on a 32-core host.
#
# usage (from exporter_models/echo, in the QNN environment of NOTES.md):
#   qnn/export_reverb_qnn.sh <out dir> [SoC ...]          default: SM8450 SM8550 SM8650 SM8750 SM8850
set -euo pipefail
OUT=${1:?usage: export_reverb_qnn.sh <out dir> [SoC ...]}; shift
mkdir -p "$OUT"
declare -A HEXAGON=([SM8450]=v69 [SM8550]=v73 [SM8650]=v75 [SM8750]=v79 [SM8850]=v81)
for SOC in "${@:-SM8450 SM8550 SM8650 SM8750 SM8850}"; do
  for S in $SOC; do
    FILE="$OUT/reverb_qnn_int4_${HEXAGON[$S]:?unknown SoC $S}.pte"
    [ -f "$FILE" ] && { echo "$S: $FILE exists"; continue; }
    echo "$(date +%T) $S -> $FILE"
    python qnn/export_qnn_v3.py "$FILE.partial" --soc "$S"
    mv "$FILE.partial" "$FILE"
  done
done
ls -l "$OUT"
