from dataclasses import dataclass

from .errors import WrapperNotFoundError

_MODEL_TYPE_WRAPPERS: dict[str, str] = {}


@dataclass
class WrapperResolution:
    wrapper: str
    reason: str
    kwargs: dict


def _is_sentence_transformer(modules_json) -> bool:
    if not modules_json:
        return False
    return any("Pooling" in m.get("type", "") for m in modules_json)


def _pooling_mode(pooling_json: dict | None) -> str:
    if pooling_json and pooling_json.get("pooling_mode_cls_token"):
        return "cls"
    return "mean"


_HANDLED_ST_SUBSTRINGS = ("Transformer", "Pooling", "Normalize")


def resolve_from_metadata(modules_json, pooling_json, model_type, architectures) -> WrapperResolution:
    if _is_sentence_transformer(modules_json):
        for m in modules_json:
            mtype = m.get("type", "")
            if not any(s in mtype for s in _HANDLED_ST_SUBSTRINGS):
                raise WrapperNotFoundError(
                    f"sentence-transformers model has unsupported module '{mtype}'; "
                    f"the sbert wrapper only handles Transformer+Pooling(+Normalize). "
                    f"Add support in wrappers/sbert.py or export this model manually."
                )
        normalize = any("Normalize" in m.get("type", "") for m in modules_json)
        return WrapperResolution(
            wrapper="sbert",
            reason="sentence-transformers metadata (modules.json Pooling module)",
            kwargs={"pooling": _pooling_mode(pooling_json), "normalize": normalize},
        )
    if model_type and model_type in _MODEL_TYPE_WRAPPERS:
        return WrapperResolution(
            wrapper=_MODEL_TYPE_WRAPPERS[model_type],
            reason=f"model_type={model_type}",
            kwargs={},
        )
    raise WrapperNotFoundError(
        f"no wrapper for model_type={model_type!r} (architectures={architectures}); "
        f"register one in exporter/wrappers/ and map it in resolver._MODEL_TYPE_WRAPPERS"
    )


def _try_hub_json(source: str, filename: str):
    import json
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(source, filename)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text())


def resolve_wrapper(source: str) -> WrapperResolution:
    from transformers import AutoConfig
    modules_json = _try_hub_json(source, "modules.json")
    pooling_json = _try_hub_json(source, "1_Pooling/config.json")
    model_type, architectures = None, None
    try:
        cfg = AutoConfig.from_pretrained(source)
        model_type = getattr(cfg, "model_type", None)
        architectures = getattr(cfg, "architectures", None)
    except Exception:
        pass
    return resolve_from_metadata(modules_json, pooling_json, model_type, architectures)
