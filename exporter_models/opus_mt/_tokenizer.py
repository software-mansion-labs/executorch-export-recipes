"""Build a HF ``tokenizer.json`` for an Opus-MT pair.

Opus-MT ships two SentencePiece models (``source.spm``, ``target.spm``) and a
shared ``vocab.json``; ``MarianTokenizer`` segments with ``source.spm`` and
maps each piece through ``vocab.json``, so the ids a model sees are NOT the
SentencePiece ids. There is no fast tokenizer upstream, and the
``Xenova/opus-mt-*`` conversions carry ``precompiled_charsmap: null``, which
HF ``tokenizers`` itself refuses to load.

This builds a Unigram model whose vocab is ``vocab.json`` in id order, scored
from ``source.spm``. Pieces that exist only on the target side get a score
below every source piece, so they are picked only where the source model
would emit ``<unk>``. Decoding needs no scores (it concatenates pieces), so
one file serves both directions.

Only components react-native-executorch's tokenizer implements are used:
Precompiled normalizer, WhitespaceSplit + Metaspace pre-tokenizer, Unigram,
TemplateProcessing, Metaspace decoder.
"""

import base64
import json


def build(source_repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    from sentencepiece import sentencepiece_model_pb2 as pb

    proto = pb.ModelProto()
    with open(hf_hub_download(source_repo, "source.spm"), "rb") as f:
        proto.ParseFromString(f.read())
    with open(hf_hub_download(source_repo, "vocab.json")) as f:
        vocab = json.load(f)
    with open(hf_hub_download(source_repo, "tokenizer_config.json")) as f:
        tok_cfg = json.load(f)
    assert proto.trainer_spec.model_type == 1, "expected a Unigram source.spm"
    assert not proto.trainer_spec.byte_fallback

    scores = {p.piece: p.score for p in proto.pieces}
    floor = min(scores.values()) - 1000.0
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in by_id] == list(range(len(by_id))), "vocab.json ids are not dense"

    # MarianTokenizer's defaults; most Opus-MT tokenizer_config.json omit them.
    eos = tok_cfg.get("eos_token", "</s>")
    unk = tok_cfg.get("unk_token", "<unk>")
    specials = [eos, unk, tok_cfg.get("pad_token", "<pad>")]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": vocab[t], "content": t, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for t in specials
        ],
        "normalizer": {
            "type": "Precompiled",
            "precompiled_charsmap": base64.b64encode(
                proto.normalizer_spec.precompiled_charsmap).decode(),
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": eos, "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": eos, "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}},
                     {"SpecialToken": {"id": eos, "type_id": 0}}],
            "special_tokens": {eos: {"id": eos, "ids": [vocab[eos]], "tokens": [eos]}},
        },
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
        "model": {
            "type": "Unigram",
            "unk_id": vocab[unk],
            "byte_fallback": False,
            "vocab": [[piece, scores.get(piece, floor)] for piece, _ in by_id],
        },
    }


def write(source_repo: str, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(source_repo), f, ensure_ascii=False)
