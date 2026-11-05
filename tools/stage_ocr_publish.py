"""Stage the OCR exports into the published HF repo layout (MODEL_SPEC convention).

The layout every `react-native-executorch-*` repo follows:

    config.json                                  {"modelName": "<repo slug>"}
    README.md
    [<size>/]<backend>/config.json               the spec'd per-backend config
    [<size>/]<backend>/<model>[_<size>]_<backend>_<precision>.pte
    <sidecars at the repo root or per size>

Per-backend `config.json` files are GENERATED from the exported `.pte` itself —
its effective model spec (MethodMeta with the `get_model_schema` overlay on top),
with every non-constant dimension written as `-1` the way config.schema.json
requires. That way the published shapes cannot drift from the artifacts.

    uv run --project exporter_models/paddleocr python tools/stage_ocr_publish.py exports exports/publish
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_schema import effective_spec  # noqa: E402

_SCHEMA_URL = (
    "https://huggingface.co/software-mansion/react-native-executorch-spec"
    "/resolve/main/config.schema.json"
)
LANGS = ["english", "latin", "japanese", "zh_sim", "korean", "telugu", "kannada", "cyrillic"]

# precision token + quantized flag per backend, from each model's NOTES.md. A
# mixed-delegate build is labelled by the precision its bulk is stored in.
MODELS = {
    "easyocr": {
        "repo": "react-native-executorch-easy-ocr",
        "model": "easy_ocr",
        "family": "easyocr",
        "capabilities": ["text-detection", "text-recognition"],
        "sizes": LANGS,
        # detect static-int8 + recognize int8 (xnn); weight-only int8 on the ANE;
        # vulkan is fp16 CRAFT on the GPU + int8 CRNN on XNNPACK, fp16-dominated.
        "precision": {"xnnpack": ("int8", True), "coreml": ("int8", True), "vulkan": ("fp16", False)},
    },
    "paddleocr": {
        "repo": "react-native-executorch-pp-ocrv6",
        "model": "pp_ocrv6",
        "family": "paddleocr",
        "capabilities": ["text-detection", "text-recognition"],
        # static int8 detect + fp32 SVTR recognize on XNNPACK; weight-only int8 on
        # the ANE; fp16 Vulkan detect + fp32 XNNPACK rec.
        "precision": {"xnnpack": ("int8", True), "coreml": ("int8", True), "vulkan": ("fp16", False)},
    },
    "paddleocr_layout": {
        "repo": "react-native-executorch-pp-doclayout-v3",
        "model": "pp_doclayout_v3",
        "family": "paddleocr",
        "capabilities": ["object-detection"],
        # int8 loses whole boxes on this RT-DETR, so nothing here is quantized.
        "precision": {"xnnpack": ("fp32", False), "coreml": ("fp16", False), "vulkan": ("fp16", False)},
    },
    "paddleocr_doc_helpers": {
        "repo": "react-native-executorch-paddle-helpers",
        "model": "paddle_helpers",
        "family": "paddleocr",
        "capabilities": ["classification"],
        # dynamic int8 (qd8) on XNNPACK, weight-only int8 on the ANE, fp16 on Vulkan.
        "precision": {"xnnpack": ("int8", True), "coreml": ("int8", True), "vulkan": ("fp16", False)},
    },
}
BACKENDS = ["xnnpack", "coreml", "vulkan"]


_MODELS_DIR = Path(__file__).resolve().parents[1] / "exporter_models"
_SCHEMA_CACHE: list = []


def _validate(config):
    """Check a generated config against the published config.schema.json."""
    import urllib.request

    import jsonschema

    if not _SCHEMA_CACHE:
        with urllib.request.urlopen(_SCHEMA_URL) as response:
            _SCHEMA_CACHE.append(json.load(response))
    jsonschema.validate(config, _SCHEMA_CACHE[0])


def _carry_over(repo: str, name: str, out_dir: Path):
    """Copy a still-current sidecar from the published repo into the staged one."""
    import urllib.request

    url = f"https://huggingface.co/software-mansion/{repo}/raw/main/{name}"
    with urllib.request.urlopen(url) as response:
        (out_dir / repo / name).write_bytes(response.read())


def _shape(param):
    """config.schema.json tensorSpec: a plain int list, -1 for a dynamic dim."""
    return [d["value"] if d["kind"] == "constant" else -1 for d in param["shape"]]


def _methods(spec):
    return {
        name: {
            "inputs": [{"shape": _shape(p), "dtype": p["dtype"]} for p in m["inputs"]],
            "outputs": [{"shape": _shape(p), "dtype": p["dtype"]} for p in m["outputs"]],
        }
        for name, m in sorted(spec.items())
    }


def stage(exports_dir: Path, out_dir: Path):
    for key, meta in MODELS.items():
        repo_dir = out_dir / meta["repo"]
        repo_dir.mkdir(parents=True, exist_ok=True)
        slug = meta["repo"].removeprefix("react-native-executorch-")
        (repo_dir / "config.json").write_text(json.dumps({"modelName": slug}, indent=2) + "\n")

        for size in meta.get("sizes", [None]):
            for backend in BACKENDS:
                precision, quantized = meta["precision"][backend]
                name = "_".join(filter(None, [meta["model"], size, backend, precision])) + ".pte"
                # The exporter writes each backend's `variants[0].file`, which is
                # this same name. Deriving the source from it instead of a second
                # hardcoded pattern means a renamed artifact fails loudly here
                # rather than silently staging a stale file left in exports/.
                src = exports_dir / key / (size or "") / name
                if not src.exists():
                    print(f"  [skip] {src} not exported")
                    continue
                dest = repo_dir / (f"{size}/{backend}" if size else backend)
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)

                config = {
                    "$schema": _SCHEMA_URL,
                    "model": meta["model"],
                    "family": meta["family"],
                    **({"size": size} if size else {}),
                    "capabilities": meta["capabilities"],
                    "backend": backend,
                    "license": "apache-2.0",
                    "variants": [
                        {
                            "file": name,
                            "precision": precision,
                            "quantized": quantized,
                            "default": True,
                            "size_bytes": src.stat().st_size,
                            "methods": _methods(effective_spec(src)),
                        }
                    ],
                }
                _validate(config)
                text = json.dumps(config, indent=2) + "\n"
                (dest / "config.json").write_text(text)
                # Keep the vendored config in step: it is the source the registry
                # resolves and the export reads `variant.file` from, so the next
                # re-export writes these exact file names.
                vendored = _MODELS_DIR / key / (f"{backend}/{size}" if size else backend)
                vendored.mkdir(parents=True, exist_ok=True)
                (vendored / "config.json").write_text(text)
                print(f"  {meta['repo']}/{dest.relative_to(repo_dir)}/{name}")

            # EasyOCR's CTC charset, one per language, beside that language's backends.
            if size:
                charset = exports_dir / f"easyocr/{size}/EasyOCR_{size}.charset.json"
                if charset.exists():
                    shutil.copy2(charset, repo_dir / size / "charset.json")

    # PP-DocLayoutV3's class names, index = the model's `classes` output value.
    labels = _MODELS_DIR / "paddleocr_layout/labels.json"
    if labels.exists():
        shutil.copy2(labels, out_dir / MODELS["paddleocr_layout"]["repo"] / "labels.json")

    # PP-OCRv6's multilingual charset is fixed by the recognizer checkpoint and is
    # unchanged by a re-export, so carry the published copy forward rather than
    # regenerating it. Named .json, not .txt: it is a JSON array of strings and
    # the client parses it as one.
    _carry_over(MODELS["paddleocr"]["repo"], "charset.json", out_dir)

    readmes = Path(__file__).resolve().parent / "ocr_readmes"
    for meta in MODELS.values():
        slug = meta["repo"].removeprefix("react-native-executorch-")
        source = readmes / f"{slug}.md"
        if source.exists():
            shutil.copy2(source, out_dir / meta["repo"] / "README.md")


if __name__ == "__main__":
    exports, out = (Path(a) for a in sys.argv[1:3])
    stage(exports, out)
    print(f"staged into {out}")
