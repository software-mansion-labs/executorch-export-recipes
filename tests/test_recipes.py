import json
from pathlib import Path

import huggingface_hub
import pytest

from exporter import cli, registry
from exporter.errors import SpecError
from exporter.runner import load_export_callable
from exporter.spec import load_spec, load_spec_file

MODELS = Path(__file__).resolve().parents[1] / "exporter_models"
CONFIGS = sorted(MODELS.rglob("config.json"))
RECIPES = {
    path.parent.relative_to(MODELS).as_posix(): json.loads(path.read_text())
    for path in sorted(MODELS.rglob("model.json"))
}
EXPORTS = [
    (model, backend, MODELS / model / (backend or "") / "export.py")
    for model, pointer in RECIPES.items()
    for backend in pointer.get("backends", [None])
]
# Legacy recipes can declare a root config in place of a pointer.
EXPORTS.extend(
    (model, None, path.with_name("export.py"))
    for path in CONFIGS
    if (model := path.parent.relative_to(MODELS).as_posix()) not in RECIPES
    and not any(parent.as_posix() in RECIPES for parent in Path(model).parents)
    and path.with_name("export.py").is_file()
)


@pytest.fixture(autouse=True)
def offline_registry(monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail(f"Unexpected Hugging Face download: {args!r} {kwargs!r}")

    monkeypatch.setattr(registry, "MODELS_DIR", MODELS)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.relative_to(MODELS).as_posix())
def test_checked_in_config(path):
    load_spec_file(path).default_variant()


def test_recipe_discovery(capsys):
    assert cli.main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["models"] == sorted(
        {model for model, _, _ in EXPORTS}
    )


@pytest.mark.parametrize(
    "model,backend,path",
    [
        pytest.param(model, backend, path, id=path.relative_to(MODELS).as_posix())
        for model, backend, path in EXPORTS
    ],
)
def test_export_script_loads(model, backend, path):
    assert registry.export_script_path(model, backend) == path
    load_export_callable(path)


@pytest.fixture
def spec_data():
    return {
        "model": "blazeface",
        "family": "blazeface",
        "capabilities": ["object-detection"],
        "backend": "xnnpack",
        "license": "apache-2.0",
        "variants": [
            {"file": "int8.pte", "precision": "int8", "quantized": True, "default": True},
            {"file": "fp32.pte", "precision": "fp32", "quantized": False, "default": True},
        ],
    }


@pytest.mark.parametrize("field", ["precision", "quantized", "default"])
def test_required_variant_field(spec_data, field):
    del spec_data["variants"][0][field]

    with pytest.raises(SpecError, match=f"missing required field '{field}'"):
        load_spec(spec_data)


@pytest.mark.parametrize("has_file,has_components", [(False, False), (True, True)])
def test_variant_requires_one_artifact_form(spec_data, has_file, has_components):
    variant = spec_data["variants"][0]
    if not has_file:
        del variant["file"]
    if has_components:
        variant["components"] = {"encoder": "encoder.pte"}

    with pytest.raises(SpecError, match="exactly one of 'file' or 'components'"):
        load_spec(spec_data)


def test_component_variant(spec_data):
    variant = spec_data["variants"][0]
    del variant["file"]
    variant["components"] = {"encoder": "encoder.pte", "decoder": "decoder.pte"}

    parsed = load_spec(spec_data).variant_by_precision("int8")
    assert parsed.file is None
    assert parsed.components == variant["components"]


@pytest.mark.parametrize(
    "flags,expected_files",
    [
        pytest.param([], ["fp32.pte"], id="default"),
        pytest.param(["--variant", "int8"], ["int8.pte"], id="explicit"),
        pytest.param(["--all-variants"], ["int8.pte", "fp32.pte"], id="all"),
    ],
)
def test_cli_variant_selection(spec_data, flags, expected_files, tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    recipe = models / "blazeface"
    script = recipe / "xnnpack" / "export.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    (recipe / "model.json").write_bytes((MODELS / "blazeface" / "model.json").read_bytes())
    monkeypatch.setattr(registry, "MODELS_DIR", models)

    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec_data))

    def download_config(repo, filename):
        assert repo == "software-mansion/react-native-executorch-blazeface"
        assert filename == "xnnpack/config.json"
        return str(config)

    def unexpected_export(*args, **kwargs):
        pytest.fail("A dry run tried to export a recipe")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_config)
    monkeypatch.setattr(cli, "export_variant", unexpected_export)
    monkeypatch.setattr(cli, "export_all", unexpected_export)
    output = tmp_path / "output"

    assert cli.main([
        "export", "blazeface", "--backend", "xnnpack", "--dry-run", "--json",
        "--output-dir", str(output), *flags,
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "model": "blazeface",
        "backend": "xnnpack",
        "export_script": str(script),
        "variants": expected_files,
        "output_dir": str(output),
        "size": None,
    }
    assert not output.exists()


def test_cli_dry_run_uses_checked_in_config(capsys):
    assert cli.main([
        "export", "facemesh", "--backend", "xnnpack", "--dry-run", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["variants"] == ["facemesh_xnnpack_fp32.pte"]
