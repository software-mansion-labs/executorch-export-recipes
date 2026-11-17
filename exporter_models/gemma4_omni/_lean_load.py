"""Loads a Gemma 4 text model without materializing it in fp32 first.

Both model ports (`_text_decoder/` and `_gemma4_edge/`) build the model at the
default fp32, read a full state dict from the safetensors checkpoint, copy it
in with `load_state_dict`, and only then cast to the export dtype. For E2B that
is ~5B parameters: ~20 GB of fp32 weights plus the state dict on top, which no
16 GB machine survives.

Inside `lean_load()`:

- `nn.Linear` / `nn.Embedding` skip their initializers. The fp32 storage they
  allocate is never written, so it is never paged in.
- `load_state_dict` assigns the checkpoint tensors instead of copying them into
  that storage, which is then freed untouched.

Peak memory is then roughly the checkpoint at its own dtype. Because nothing was
initialized, a parameter the checkpoint does not provide would keep garbage, so
any missing key other than the expected tied `lm_head.weight` is an error.
"""

import contextlib

import torch.nn as nn

_EXPECTED_MISSING = {"lm_head.weight"}


@contextlib.contextmanager
def lean_load():
    saved = (
        nn.Linear.reset_parameters,
        nn.Embedding.reset_parameters,
        nn.Module.load_state_dict,
    )
    orig_load = nn.Module.load_state_dict

    def load_by_assignment(self, state_dict, strict=True, assign=False):
        result = orig_load(self, state_dict, strict=strict, assign=True)
        # Buffers (KV caches, RoPE tables) are built by the model, not loaded;
        # only parameters can be left holding uninitialized memory.
        params = {name for name, _ in self.named_parameters()}
        missing = (set(result.missing_keys) & params) - _EXPECTED_MISSING
        if missing:
            raise RuntimeError(
                f"lean_load: {len(missing)} parameter(s) missing from the checkpoint "
                f"would stay uninitialized, e.g. {sorted(missing)[:5]}"
            )
        return result

    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    nn.Module.load_state_dict = load_by_assignment
    try:
        yield
    finally:
        (
            nn.Linear.reset_parameters,
            nn.Embedding.reset_parameters,
            nn.Module.load_state_dict,
        ) = saved
