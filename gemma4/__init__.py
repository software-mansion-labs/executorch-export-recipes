"""Gemma4 ExecuTorch export package.

Self-contained model definition plus four export recipes (XNNPACK and Vulkan,
text-only and multimodal). Extracted from `gemma_export/` on 2026-07-30; see
`docs/RECIPES.md` for the provenance of each recipe and `docs/LEVERS.md` for
the optimisation levers that were confirmed and falsified along the way.

Import-order note: any module that touches ExecuTorch runtime kernels must
import `executorch.runtime.Runtime` BEFORE `executorch.kernels.quantized`.
`gemma4.export` enforces this at the top of its `__main__`; if you build a
pipeline by hand, do the same or `embedding_{2,4}bit.out` will fail to
resolve at `load_method` time.
"""

__all__ = ["model", "export"]
