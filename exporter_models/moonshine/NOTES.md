# moonshine notes

Moonshine-tiny STT, 2-method (encoder + decoder). Keras→pytorch conversion was
done upstream; `moonshine.pth` is the converted weights, `_model_pt.py` is the
pure-torch model (faithful copy of the recipe's `model_pt.py`). `export.py`
transcribes `exports/moonshine/export_et.py` (same wrappers, dynamic shapes,
`ExecutorchBackendConfig(passes=[])`).

## config.json uses the exported dynamic-MAX shapes (intentional)

The encoder/decoder export with dynamic seq dims (encoder dim1 207..1250 ×384;
decoder seq 2..178, enc 10..1248). The exported `.pte`'s method_meta reports the
dynamic dim's UPPER BOUND, so the local `config.json` declares the static-max
shapes the runtime actually reports:
- encoder `forward`: `[1,480000] → [1,1248,288]`
- decoder `forward`: `seq[1,178], enc[1,1248,288] → [1,179]`

This matches the real export (verified `io_ok` + `parity_ok`) — see commit
`98811b7`. Local config (not a model.json pointer) because moonshine's HF repo has
no spec config, only a `{modelName}` stub (hand-authored, like kokoro).

## Core ML / MLX: neither backend, by decision

This model ships **XNNPACK only** -- no Core ML and no MLX export exists, and neither
has been attempted.

This is a prioritisation decision, not a technical finding: during the 2026-08 Apple
backend sweep this model was explicitly deprioritised in favour of the higher-traffic
computer-vision and embedding models. Nothing is known to block a Core ML or MLX
export here. If you want one, start from a sibling model's `coreml/export.py` --
there is no recorded failure to work around.
