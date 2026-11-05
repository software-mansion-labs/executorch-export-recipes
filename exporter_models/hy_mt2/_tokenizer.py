"""Write the tokenizer files react-native-executorch loads for Hy-MT2.

    python exporter_models/hy_mt2/_tokenizer.py <out_dir>

writes ``tokenizer.json`` and ``tokenizer_config.json`` next to the size's
backend dirs on HF (``1_8b/``).

``tokenizer.json`` is upstream's with three rewrites that leave every id
unchanged, because the HFTokenizer in RNE's native libs (0.10.7) misreads two
of its components (both fixed upstream in meta-pytorch/tokenizers#225):

- the empty ``Sequence`` normalizer becomes ``null``. An empty Sequence is a
  no-op, but the native loader rejects it.
- each ``Split`` (Regex, ``Isolated``, ``invert: false``) gets an alternative
  that matches the text between its matches. ``Isolated`` keeps both, the
  native loader keeps only the matches, so the digit and CJK splits dropped
  all other text, and the last split dropped bare digit runs.

``tokenizer_config.json`` is upstream's plus ``chat_template``, which upstream
ships as a separate ``chat_template.jinja`` and RNE reads from the config.
"""

import json
import sys
from pathlib import Path

SOURCE = "tencent/Hy-MT2-1.8B"

_CJK = "一-龥぀-ゟ゠-ヿ"

# Upstream pattern -> the same pattern plus "the text between matches". For the
# last one that text is a run of characters none of its branches can start
# at: not whitespace, letter, mark, punctuation or symbol, and not a prefix
# directly followed by a letter or mark.
_TILING = {
    r"\p{N}{1,3}": r"\p{N}{1,3}|\P{N}+",
    f"[{_CJK}]+": f"[{_CJK}]+|[^{_CJK}]+",
}
_GAP = r"|(?:[^\s\p{L}\p{M}\p{P}\p{S}](?![\p{L}\p{M}]))+"


def build(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    tok = json.loads(Path(hf_hub_download(SOURCE, "tokenizer.json")).read_text())
    assert tok["normalizer"] == {"type": "Sequence", "normalizers": []}
    tok["normalizer"] = None

    splits = tok["pre_tokenizer"]["pretokenizers"][:3]
    assert [s["type"] for s in splits] == ["Split"] * 3
    for s in splits:
        assert s["behavior"] == "Isolated" and not s["invert"]
    for s in splits[:2]:
        s["pattern"]["Regex"] = _TILING[s["pattern"]["Regex"]]
    splits[2]["pattern"]["Regex"] += _GAP

    cfg = json.loads(Path(hf_hub_download(SOURCE, "tokenizer_config.json")).read_text())
    assert "chat_template" not in cfg
    cfg["chat_template"] = Path(hf_hub_download(SOURCE, "chat_template.jinja")).read_text()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
