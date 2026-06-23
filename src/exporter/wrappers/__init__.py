from collections.abc import Callable

from ..errors import WrapperNotFoundError

_BUILDERS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        _BUILDERS[name] = fn
        return fn
    return deco


def get_builder(name: str) -> Callable:
    if name not in _BUILDERS:
        raise WrapperNotFoundError(
            f"no wrapper '{name}'; register one in exporter/wrappers/ (have: {', '.join(sorted(_BUILDERS)) or 'none'})"
        )
    return _BUILDERS[name]


def available() -> list[str]:
    return sorted(_BUILDERS)
