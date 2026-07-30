"""Gemma4 export recipes.

Typical use is the CLI:

    PYTHONPATH=. python -m gemma4.export --recipe vulkan-mm --dry-run

Programmatic use:

    from gemma4.export import presets, validate, pipeline

    recipe = presets.get("xnnpack-mm").replace_nested("lengths", max_seq_len=4096)
    validate.assert_valid(recipe)          # raises on any error, returns warnings
    pipeline.export(recipe, checkpoint=..., out_path=...)

Module map:
    config      typed Recipe / QuantSpec / LengthSpec / AttnFlags
    presets     the four recipes, with provenance and trust level
    validate    pre-flight rules; every one encodes a real failure
    patches     gated monkey-patches applied before export
    quant       quantisation helpers + the Vulkan fp16 retarget
    wrappers    per-method export nn.Modules
    pipeline    build -> transform -> quantize -> export -> lower -> write
"""

__all__ = [
    "config",
    "presets",
    "validate",
    "patches",
    "quant",
    "wrappers",
    "pipeline",
]
