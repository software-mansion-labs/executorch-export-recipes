"""Bridge between the exporter harness's config dict and the edge `Recipe`.

Added when this tree was vendored into `exporter_models/`; not part of the
original `gemma4/` package.

`skills/model-conventions` makes the model's `config.json` the single source of
truth for LLM runner constants (`get_bos_id`, `get_eos_ids`, `get_max_seq_len`,
`has_ple`, the `audio_*` block, ...). The stock `Recipe.constant_methods()`
DERIVES those from the recipe instead, which would let the `.pte` and the
published contract drift apart silently.

`HarnessRecipe` overrides that one method to return what the config declares,
and `check_constants` asserts the two agree on the values that are also baked
into the graph — so a config edit that the recipe constants do not follow fails
before a multi-GB export rather than after it.
"""
from __future__ import annotations

import dataclasses
import glob
import logging
import os

from .config import Recipe

__all__ = ["HarnessRecipe", "check_constants", "resolve_checkpoint"]

logger = logging.getLogger(__name__)

# The public Gemma 4 E2B checkpoint. Hardcoded here rather than read from the
# model config, per the convention that vendored code owns its source id.
_HF_REPO_ID = "google/gemma-4-E2B-it"

# Where `snapshot_download` parks that repo. Derived from the repo id so the
# two cannot drift: HF maps "org/name" to "models--org--name" in the cache.
# Matches ANY *.safetensors, not just `model.safetensors`, because a sharded
# checkpoint is legal -- `pipeline._resolve_checkpoint` accepts either.
_HF_CACHE_GLOB = (
    "~/.cache/huggingface/hub/models--"
    + _HF_REPO_ID.replace("/", "--")
    + "/snapshots/*/*.safetensors"
)

# What the export actually needs: weights + config, plus the tokenizer files a
# runner will want beside them. Skips any .pte / .gguf mirrors the repo also
# publishes, which would otherwise double the download.
_DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.model"]


def _download_checkpoint() -> str:
    """Fetch `_HF_REPO_ID` into the HF cache and return its snapshot directory.

    Only called when nothing was found locally. Set `$HF_HUB_OFFLINE=1` to make
    this fail fast instead of downloading.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "no Gemma 4 checkpoint found locally, and auto-download needs "
            "`huggingface_hub`, which is not installed. Install it, or set "
            "$GEMMA4_CHECKPOINT to an existing checkpoint directory."
        ) from e

    logger.warning(
        "no Gemma 4 checkpoint found locally; downloading %r from HuggingFace "
        "(several GB, one time). Set $GEMMA4_CHECKPOINT to use a local "
        "checkpoint instead, or $HF_HUB_OFFLINE=1 to refuse the download.",
        _HF_REPO_ID,
    )
    try:
        path = snapshot_download(_HF_REPO_ID, allow_patterns=_DOWNLOAD_PATTERNS)
    except Exception as e:  # noqa: BLE001 -- surfaced verbatim below
        raise SystemExit(
            f"could not download {_HF_REPO_ID!r} from HuggingFace: {e}\n"
            "If the repo is gated, accept its licence on the model page and "
            "authenticate (`hf auth login`, or `huggingface-cli login` on older "
            "huggingface_hub). Otherwise set $GEMMA4_CHECKPOINT to a local "
            "checkpoint directory containing *.safetensors and config.json."
        ) from e

    if not glob.glob(os.path.join(path, "*.safetensors")):
        raise SystemExit(
            f"downloaded {_HF_REPO_ID!r} to {path}, but it contains no "
            "*.safetensors. Set $GEMMA4_CHECKPOINT to a checkpoint directory "
            "that does."
        )
    logger.info("checkpoint ready at %s", path)
    return path


def resolve_checkpoint() -> str:
    """Return the HF model directory holding `*.safetensors` + `config.json`.

    `$GEMMA4_CHECKPOINT` wins, then the HF cache, then a download from the Hub.
    The harness `export()` entry point takes no arguments, so there is no
    `--checkpoint` flag to fall back on.

    NOTE the tokenizer in this directory must be the LOCAL Gemma4 fork, not
    upstream `gemma3n` -- the placeholder ids differ (image 258880 / audio
    258881 here vs 262145 / 262273 upstream). Export does not read the
    tokenizer, but a runner driving the resulting .pte does. If the auto-
    downloaded repo turns out to be the upstream lineage, point
    $GEMMA4_CHECKPOINT at the fork instead.
    """
    env = os.environ.get("GEMMA4_CHECKPOINT")
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.exists(path):
            raise SystemExit(f"GEMMA4_CHECKPOINT is set but does not exist: {path}")
        return path
    hits = glob.glob(os.path.expanduser(_HF_CACHE_GLOB))
    if hits:
        return os.path.dirname(hits[0])
    return _download_checkpoint()


@dataclasses.dataclass(frozen=True)
class HarnessRecipe(Recipe):
    """A `Recipe` whose `constant_methods()` come from `config.json`.

    `dataclasses.replace` (and therefore `Recipe.overridden` /
    `Recipe.replace_nested`) preserves the subclass, so the override survives
    any further tweaking.
    """

    constants: dict = dataclasses.field(default_factory=dict)

    def constant_methods(self) -> dict:
        return dict(self.constants)


# Constants that are ALSO baked into the traced graph, and so cannot disagree
# with the recipe. Everything else in the config (token ids, placeholder ids,
# the audio sample-rate) is metadata the graph never sees, and the config wins
# outright.
def _graph_bound_expectations(recipe: Recipe) -> dict:
    L = recipe.lengths
    expected = {
        # The runner reads this as the PER-CALL PREFILL CHUNK SIZE, not the
        # context limit -- it must equal the seq bound the graph accepts, or a
        # prefill call overruns the arena with no error (rule R8).
        "get_max_seq_len": L.effective_text_decoder_prefill_len,
        "get_max_context_len": L.max_seq_len,
        "get_vocab_size": recipe.vocab_size,
        "get_bos_id": recipe.bos_id,
        "get_eos_ids": list(recipe.eos_ids),
    }
    if recipe.modality == "multimodal":
        expected.update(
            {
                "image_placeholder_id": recipe.image_placeholder_id,
                "audio_placeholder_id": recipe.audio_placeholder_id,
                "audio_soft_tokens_max": L.audio_soft_tokens_max,
                "audio_block_samples": L.audio_block_samples,
                "audio_block_k_min": L.audio_k_min,
                "audio_block_k_max": L.audio_k_max,
                "audio_sample_count_min": L.audio_sample_count_min,
                "audio_sample_count_max": L.audio_sample_count_max,
            }
        )
    return expected


def check_constants(recipe: Recipe) -> None:
    """Raise if the config's constants contradict the baked recipe.

    Only keys present in BOTH are compared, so a config may legitimately omit
    any of them; it may not declare one and mean something else by it.

    There is deliberately NO deviation allowlist. Every declared value is
    written into the .pte's `constant_methods` verbatim, so tolerating a
    disagreement ships an artifact whose metadata lies about its own graph: a
    `get_max_seq_len` above the traced seq bound makes a runner overrun the
    arena, and a wrong `image_placeholder_id` makes it splice image tokens at a
    position the tokenizer never emits. When this raises, one of the two is
    wrong -- fix it.
    """
    declared = recipe.constant_methods()
    if not declared:
        raise ValueError(
            "the harness config declares no LLM runner constants; expected "
            "get_bos_id / get_eos_ids / get_max_seq_len / ... as non-IO method "
            "entries in the variant's 'methods' block"
        )
    mismatched = {
        key: (declared[key], want)
        for key, want in _graph_bound_expectations(recipe).items()
        if key in declared and declared[key] != want
    }

    if mismatched:
        detail = "; ".join(
            f"{k}: config says {got!r}, graph is built for {want!r}"
            for k, (got, want) in sorted(mismatched.items())
        )
        raise ValueError(
            f"config.json constants disagree with the {recipe.name!r} recipe -- "
            f"{detail}. Fix whichever is wrong; do not export a .pte whose "
            f"metadata lies about its own graph."
        )
