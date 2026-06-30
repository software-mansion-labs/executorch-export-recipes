from dataclasses import dataclass
from pathlib import Path

from .runner import run_export
from .spec import Spec, Variant


@dataclass
class VariantResult:
    file: str
    precision: str


@dataclass
class ExportPlan:
    model: str
    backend: str
    export_script: str
    variants: list[str]
    output_dir: str


def _select(spec: Spec, which: str | None) -> list[Variant]:
    if which is None:
        return [spec.default_variant()]
    if which == "all":
        return list(spec.variants)
    return [spec.variant_by_precision(which)]


def plan(spec: Spec, *, export_script: str, which: str | None, output_dir: str) -> ExportPlan:
    variants = _select(spec, which)
    return ExportPlan(
        model=spec.model,
        backend=spec.backend,
        export_script=str(export_script),
        variants=[v.file for v in variants],
        output_dir=output_dir,
    )


def _variant_config(variant: Variant) -> dict:
    return {
        "file": variant.file,
        "precision": variant.precision,
        "quantized": variant.quantized,
        "methods": {
            name: {
                "inputs": [{"shape": t.shape, "dtype": t.dtype} for t in m.inputs],
                "outputs": [{"shape": t.shape, "dtype": t.dtype} for t in m.outputs],
            }
            for name, m in variant.methods.items()
        },
        "constants": dict(variant.constants),
    }


def export_variant(spec: Spec, variant: Variant, *, export_script: str, output_dir: str, size: str | None = None) -> VariantResult:
    config_dict = {
        "model": spec.model,
        "backend": spec.backend,
        "size": size,
        "variant": _variant_config(variant),
    }
    run_export(export_script, config_dict, output_dir)
    return VariantResult(file=variant.file, precision=variant.precision)


def export_all(spec: Spec, *, export_script: str, output_dir: str, size: str | None = None) -> list[VariantResult]:
    config_dict = {
        "model": spec.model,
        "backend": spec.backend,
        "size": size,
        "variants": [_variant_config(v) for v in spec.variants],
    }
    run_export(export_script, config_dict, output_dir)
    return [VariantResult(file=v.file, precision=v.precision) for v in spec.variants]
