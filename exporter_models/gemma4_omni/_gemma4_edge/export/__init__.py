"""Gemma4 edge export machinery.

Driven by the harness, not by a CLI: each `<backend>/export.py` beside this
package declares its recipe as module-level constants, builds a `Recipe` from
them, and calls `pipeline.export(...)`.

    from _gemma4_edge.export import config, pipeline

    recipe = config.Recipe(name="vulkan-text", backend="vulkan", ...)
    pipeline.export(recipe, checkpoint=..., out_path=...)

Module map:
    config      typed Recipe / QuantSpec / LengthSpec / AttnFlags
    patches     gated monkey-patches applied before export
    quant       quantisation helpers + the Vulkan fp16 retarget
    wrappers    per-method export nn.Modules
    pipeline    build -> transform -> quantize -> export -> lower -> write

Dropped when this tree was vendored into the harness: `presets.py` (the four
recipes now live as constants in each `export.py`), `validate.py` (its rules
are recorded as prose in the model's NOTES.md) and `__main__.py` (the CLI is
`exporter export`). See NOTES.md.
"""

__all__ = [
    "config",
    "harness",
    "patches",
    "quant",
    "wrappers",
    "pipeline",
]
