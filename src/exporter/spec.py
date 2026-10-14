import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SpecError

_CAPABILITIES = frozenset({
    "text-generation", "vision", "speech-to-text", "text-to-speech",
    "classification", "object-detection", "semantic-segmentation",
    "instance-segmentation", "style-transfer", "text-embedding",
    "image-embedding", "image-generation", "voice-activity-detection",
    "text-detection", "text-recognition",
    "image-text-to-text", "audio-text-to-text",
})
_BACKENDS = frozenset({"xnnpack", "coreml", "mlx", "vulkan"})


@dataclass
class IOTensor:
    shape: list[int]
    dtype: str
    name: str | None = None


@dataclass
class Method:
    name: str
    inputs: list[IOTensor]
    outputs: list[IOTensor]


@dataclass
class Variant:
    file: str | None
    precision: str
    quantized: bool
    default: bool
    methods: dict[str, Method] = field(default_factory=dict)
    components: dict[str, str] | None = None
    # Post-conversion weight compression, currently Core ML only: "int8" applies
    # op_linear_quantizer_config per-channel. None leaves weights at the compute
    # precision. Distinct from `precision`, which is the compute precision.
    quantize: str | None = None
    # Core ML only: which compute units the delegate may dispatch to
    # ("ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"). None leaves the choice
    # to the export script's own default. This is a correctness knob as much as
    # a performance one: some graphs are wrong on the ANE or on CPU-only fp16.
    compute_unit: str | None = None
    # LLM runner constants declared in the config as non-IO method entries
    # (get_bos_id, get_eos_ids, get_max_seq_len, use_kv_cache, ...). Values are
    # int / list / bool / null. Embedded into the .pte at export time.
    constants: dict = field(default_factory=dict)


@dataclass
class Spec:
    model: str
    family: str
    capabilities: list[str]
    backend: str
    license: str
    variants: list[Variant]
    schema: str | None = None
    size: str | None = None

    def default_variant(self) -> Variant:
        defaults = [v for v in self.variants if v.default]
        if not defaults:
            raise SpecError(f"model '{self.model}' has no variant marked default:true")
        # Canonical schema allows one default per quant-group (one quantized, one not).
        # Prefer the non-quantized default as the overall default; fall back to a
        # quantized default when the model ships only quantized variants.
        non_quant = [v for v in defaults if not v.quantized]
        if non_quant:
            return non_quant[0]
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
        out.append(IOTensor(shape=list(t["shape"]), dtype=str(t["dtype"]), name=t.get("name")))
    return out


def load_spec(data: dict) -> Spec:
    model = _require(data, "model", str)
    family = _require(data, "family", str)
    capabilities = _require(data, "capabilities", list)
    backend = _require(data, "backend", str)
    license_ = _require(data, "license", str)
    raw_variants = _require(data, "variants", list)

    bad_caps = [c for c in capabilities if c not in _CAPABILITIES]
    if bad_caps:
        raise SpecError(f"unknown capabilities {bad_caps}; must be in the canonical enum")
    if backend not in _BACKENDS:
        raise SpecError(f"unknown backend '{backend}' (must be one of {sorted(_BACKENDS)})")
    if not raw_variants:
        raise SpecError(f"model '{model}' has no variants")

    variants = []
    for rv in raw_variants:
        has_file = "file" in rv and rv["file"] is not None
        has_components = "components" in rv and rv["components"] is not None
        if has_file == has_components:
            raise SpecError(
                f"model '{model}' variant must have exactly one of 'file' or 'components'"
            )
        methods = {}
        constants = {}
        for mname, m in rv.get("methods", {}).items():
            if isinstance(m, dict) and "inputs" in m and "outputs" in m:
                methods[mname] = Method(name=mname, inputs=_io(m["inputs"]), outputs=_io(m["outputs"]))
            else:
                # LLM runner constant (int/list/bool/null, or dict w/o IO)
                constants[mname] = m
        variants.append(
            Variant(
                file=rv.get("file"),
                precision=_require(rv, "precision", str),
                quantized=_require(rv, "quantized", bool),
                default=_require(rv, "default", bool),
                methods=methods,
                components=rv.get("components"),
                quantize=rv.get("quantize"),
                compute_unit=rv.get("compute_unit"),
                constants=constants,
            )
        )

    return Spec(
        model=model,
        family=family,
        capabilities=list(capabilities),
        backend=backend,
        license=license_,
        variants=variants,
        schema=data.get("$schema"),
        size=data.get("size"),
    )


def load_spec_file(path) -> Spec:
    path = Path(path)
    if not path.exists():
        raise SpecError(f"spec file not found: {path}")
    return load_spec(json.loads(path.read_text()))
