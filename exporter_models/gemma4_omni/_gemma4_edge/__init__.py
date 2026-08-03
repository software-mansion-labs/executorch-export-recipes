"""Gemma 4 E2B edge port — model definition and ExecuTorch export machinery.

Vendored per the harness convention that each model owns its code. The tree is
byte-identical in `exporter_models/gemma4/` and `exporter_models/gemma4_omni/`;
the text model does not use `model.vision` / `model.audio` / `model.mel_frontend`,
but they are kept so `export/pipeline.py` and `export/wrappers.py` — which
import them at module scope — stay unmodified across both copies.

This is a DIFFERENT port from the `_text_decoder/` (gemma4) and `_gemma4/`
(gemma4_omni) trees that back the MLX backend: different class hierarchy,
different state-dict layout, not interchangeable. See NOTES.md.

Load it by path with a `sys.modules` key unique to the model directory
(`gemma4_edge_text` / `gemma4_edge_omni`) — `export/patches.py` keeps its
applied-set in module globals, and two copies sharing a key would let a
wrapping patch nest inside itself.
"""

__all__ = ["model", "export"]
