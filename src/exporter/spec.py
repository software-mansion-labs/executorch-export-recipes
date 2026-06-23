import json
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecError


@dataclass
class IOTensor:
    shape: list[int]
    dtype: str


@dataclass
class Method:
    name: str
    inputs: list[IOTensor]
    outputs: list[IOTensor]


@dataclass
class Variant:
    file: str
    precision: str
    quantized: bool
    default: bool
    methods: dict[str, Method]


@dataclass
class Spec:
    source: str
    model: str
    capabilities: list[str]
    backend: str
    license: str
    variants: list[Variant]
    schema: str | None = None

    def default_variant(self) -> Variant:
        defaults = [v for v in self.variants if v.default]
        if len(defaults) == 0:
            raise SpecError(f"model '{self.model}' has no variant marked default:true")
        if len(defaults) > 1:
            raise SpecError(f"model '{self.model}' has multiple default variants")
        return defaults[0]

    def variant_by_precision(self, precision: str) -> Variant:
        for v in self.variants:
            if v.precision == precision:
                return v
        available = ", ".join(v.precision for v in self.variants)
        raise SpecError(
            f"no variant with precision '{precision}' in '{self.model}' (have: {available})"
        )


def _require(data: dict, key: str, typ: type):
    if key not in data:
        raise SpecError(f"spec missing required field '{key}'")
    if not isinstance(data[key], typ):
        raise SpecError(f"spec field '{key}' must be {typ.__name__}, got {type(data[key]).__name__}")
    return data[key]


def _io(items: list) -> list[IOTensor]:
    out = []
    for t in items:
        if "shape" not in t or "dtype" not in t:
            raise SpecError(f"tensor entry must have 'shape' and 'dtype', got {t}")
        out.append(IOTensor(shape=list(t["shape"]), dtype=str(t["dtype"])))
    return out


def load_spec(data: dict) -> Spec:
    source = _require(data, "source", str)
    model = _require(data, "model", str)
    capabilities = _require(data, "capabilities", list)
    backend = _require(data, "backend", str)
    license_ = _require(data, "license", str)
    raw_variants = _require(data, "variants", list)
    if not raw_variants:
        raise SpecError(f"model '{model}' has no variants")

    variants = []
    for rv in raw_variants:
        methods = {}
        for mname, m in _require(rv, "methods", dict).items():
            if "inputs" not in m or "outputs" not in m:
                raise SpecError(f"method '{mname}' must have 'inputs' and 'outputs'")
            methods[mname] = Method(
                name=mname,
                inputs=_io(m["inputs"]),
                outputs=_io(m["outputs"]),
            )
        variants.append(
            Variant(
                file=_require(rv, "file", str),
                precision=_require(rv, "precision", str),
                quantized=_require(rv, "quantized", bool),
                default=_require(rv, "default", bool),
                methods=methods,
            )
        )

    return Spec(
        source=source,
        model=model,
        capabilities=list(capabilities),
        backend=backend,
        license=license_,
        variants=variants,
        schema=data.get("$schema"),
    )


def load_spec_file(path) -> Spec:
    path = Path(path)
    if not path.exists():
        raise SpecError(f"spec file not found: {path}")
    return load_spec(json.loads(path.read_text()))
