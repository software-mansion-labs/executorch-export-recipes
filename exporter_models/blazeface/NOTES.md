# blazeface notes

## Core ML / MLX: neither backend, by decision

This model ships **XNNPACK only** -- no Core ML and no MLX export exists, and neither
has been attempted.

This is a prioritisation decision, not a technical finding: during the 2026-08 Apple
backend sweep this model was explicitly deprioritised in favour of the higher-traffic
computer-vision and embedding models. Nothing is known to block a Core ML or MLX
export here. If you want one, start from a sibling model's `coreml/export.py` --
there is no recorded failure to work around.
