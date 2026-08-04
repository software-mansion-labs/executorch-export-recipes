import json
from pathlib import Path
import jsonschema

_ROOT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "model.schema.json"


def load_and_validate_schema(
    schema_path: Path | str, replacements: dict[str, str] | None = None
) -> str:
    """Read a model schema JSON file, validate it inline against model.schema.json,
    and return the validated JSON string.

    Args:
        schema_path: Path to the model's schema.json file.
        replacements: Optional template string replacements (e.g. `{"n_vocab": "51865"}`).

    Returns:
        Validated JSON string.
    """
    text = Path(schema_path).read_text()
    if replacements:
        for k, v in replacements.items():
            text = text.replace(f"{{{k}}}", v)

    data = json.loads(text)

    root_schema = json.loads(_ROOT_SCHEMA_PATH.read_text())
    jsonschema.validate(instance=data, schema=root_schema)

    return json.dumps(data)
