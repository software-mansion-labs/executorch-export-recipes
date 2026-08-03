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
_HF_CACHE_GLOB = (
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/*/model.safetensors"
)


def resolve_checkpoint() -> str:
    """Return the HF model directory holding `model.safetensors` + `config.json`.

    `$GEMMA4_CHECKPOINT` wins, then the HF cache. The harness `export()` entry
    point takes no arguments, so there is no `--checkpoint` flag to fall back on.

    NOTE the tokenizer in this directory must be the LOCAL Gemma4 fork, not
    upstream `gemma3n` -- the placeholder ids differ (image 258880 / audio
    258881 here vs 262145 / 262273 upstream). Export does not read the
    tokenizer, but a runner driving the resulting .pte does.
    """
    env = os.environ.get("GEMMA4_CHECKPOINT")
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.exists(path):
            raise SystemExit(f"GEMMA4_CHECKPOINT is set but does not exist: {path}")
        return path
    hits = glob.glob(os.path.expanduser(_HF_CACHE_GLOB))
    if not hits:
        raise SystemExit(
            "Gemma 4 needs a checkpoint directory containing model.safetensors "
            "and config.json. None found in the HF cache; set $GEMMA4_CHECKPOINT."
        )
    return os.path.dirname(hits[0])


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


def check_constants(recipe: Recipe, known_deviations=frozenset()) -> None:
    """Raise if the config's constants contradict the baked recipe.

    Only keys present in BOTH are compared, so a config may legitimately omit
    any of them; it may not declare one and mean something else by it.

    `known_deviations` names keys where the published config is KNOWN to
    disagree with the recipe and the config nevertheless wins (this repo's
    configs are verbatim copies of the published ones). Those are logged at
    WARNING and allowed; any OTHER disagreement still raises, so accidental
    drift is still caught. Each caller must justify every key it lists.

    Note what an allowed deviation means in practice: the value goes into the
    .pte's `constant_methods`, so the artifact's metadata will describe
    something the graph does not do. That is a deliberate choice to mirror the
    published contract, not a safe default -- see the model's NOTES.md.
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

    for key in sorted(set(mismatched) & set(known_deviations)):
        got, want = mismatched.pop(key)
        logger.warning(
            "%s: config.json declares %s=%r but the graph is built for %r. "
            "Known published-config deviation, allowed on purpose -- the .pte's "
            "metadata will carry the config's value. See NOTES.md.",
            recipe.name, key, got, want,
        )

    unused = sorted(set(known_deviations) - set(_graph_bound_expectations(recipe)))
    if unused:
        logger.warning(
            "%s: known_deviations lists %s, which are not checked constants -- "
            "stale entry?", recipe.name, unused,
        )

    if mismatched:
        detail = "; ".join(
            f"{k}: config says {got!r}, graph is built for {want!r}"
            for k, (got, want) in sorted(mismatched.items())
        )
        raise ValueError(
            f"config.json constants disagree with the {recipe.name!r} recipe -- "
            f"{detail}. Fix whichever is wrong, or add the key to this export's "
            f"KNOWN_CONFIG_DEVIATIONS with a justification; do not silently "
            f"export a .pte whose metadata lies about its own graph."
        )
