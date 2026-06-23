from .resolver import resolve_wrapper
from .spec import IOTensor, Method, Variant
from .wrappers import get_builder

_SCHEMA = "https://huggingface.co/software-mansion/react-native-executorch-spec/resolve/main/config.schema.json"

_CAPS = {"sbert": ["text-embedding"]}


def _default_model_name(source: str) -> str:
    name = source.split("/")[-1]
    return name.replace("-", "_").lower()


def _capabilities_for(wrapper: str) -> list[str]:
    return _CAPS.get(wrapper, [])


def draft_spec(source: str, *, model: str | None = None, backend: str = "xnnpack", max_seq_len: int = 254) -> dict:
    import torch

    res = resolve_wrapper(source)
    model = model or _default_model_name(source)

    fwd = Method(
        "forward",
        inputs=[IOTensor([1, max_seq_len], "int64"), IOTensor([1, max_seq_len], "int64")],
        outputs=[],
    )
    probe_variant = Variant(file="", precision="fp32", quantized=False, default=True, methods={"forward": fwd})
    module = get_builder(res.wrapper)(source, probe_variant, **res.kwargs)
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(1, max_seq_len, dtype=torch.int64),
                     torch.ones(1, max_seq_len, dtype=torch.int64))
    out_shape = list(out.shape)
    out_dtype = str(out.dtype).replace("torch.", "")

    file_name = f"{model}_{backend}_fp32.pte"
    return {
        "$schema": _SCHEMA,
        "source": source,
        "model": model,
        "capabilities": _capabilities_for(res.wrapper),
        "backend": backend,
        "license": "unknown",
        "variants": [
            {
                "file": file_name,
                "precision": "fp32",
                "quantized": False,
                "default": True,
                "methods": {
                    "forward": {
                        "inputs": [
                            {"shape": [1, max_seq_len], "dtype": "int64"},
                            {"shape": [1, max_seq_len], "dtype": "int64"},
                        ],
                        "outputs": [{"shape": out_shape, "dtype": out_dtype}],
                    }
                },
            }
        ],
    }
