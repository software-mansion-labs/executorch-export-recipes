import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

# Meta-format checkpoints converted from the upstream HF weights, keyed by
# source repo so each size converts once per machine. Shared with the XNNPACK
# export: the checkpoint is backend-independent.
_CKPT_CACHE = Path.home() / ".cache" / "exporter_meta_checkpoints"

# What the HF->Meta converter needs to see in the snapshot.
_WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]


def _load_params():
    name = "eurollm_vulkan_params"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(source_repo: str) -> str:
    """Download the upstream HF weights and convert them to Meta format."""
    # qwen3's converter is a plain Llama key map that keeps an untied lm_head.
    from executorch.examples.models.qwen3.convert_weights import convert_weights
    from huggingface_hub import snapshot_download

    _CKPT_CACHE.mkdir(parents=True, exist_ok=True)
    out = _CKPT_CACHE / f"{source_repo.replace('/', '_')}.pth"
    if out.exists():
        return str(out)
    src = snapshot_download(source_repo, allow_patterns=_WEIGHT_PATTERNS)
    # Some converters write scratch files into the cwd; keep them out of the repo.
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        convert_weights(src, str(out))
    finally:
        os.chdir(cwd)
    return str(out)


def _vulkan_partitioner(*_args, **_kwargs):
    """A VulkanPartitioner that claims the whole decoder, attention included.

    Leaving `llama::custom_sdpa` / `llama::update_cache` on the host splits the
    decoder into 57 subgraphs -- two delegate calls plus three portable kernels
    per layer -- and each boundary costs a staging copy and a queue submission.
    Letting Vulkan's `causal_sdpa` pattern fuse them back into
    `sdpa_with_kv_cache` collapses the model to a SINGLE delegate, which on a
    Galaxy S26 Ultra is worth 2.1x on decode (41.5 -> 86.2 tok/s) and drops RSS
    from 768 to 300 MiB, because the KV cache then lives in GPU memory instead
    of the host arena.

    `force_fp16` runs the delegate in half precision. It leaves the generated
    text unchanged on the prompts tested but halves the KV cache traffic, which
    is worth a further ~15% on decode once the context is a couple of hundred
    tokens deep (61.4 -> 70.3 tok/s at 206 tokens).
    """
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    return VulkanPartitioner(
        compile_options={"require_dynamic_shapes": False, "force_fp16": True}
    )


def export(config: dict, out_dir: str) -> str:
    from executorch.examples.models.llama.export_llama_lib import export_llama
    from executorch.extension.llm.export.config.llm_config import (
        DtypeOverride, LlmConfig, ModelType,
    )

    p = _load_params()
    size = config.get("size") or "1_7b"
    spec = p.SIZES[size]
    variant = config["variant"]

    # Resolve before the chdir below: export_llama writes output_name relative
    # to the cwd, so a relative --output-dir would land in the temp dir.
    out_dir = Path(out_dir).resolve()
    out_path = str(out_dir / variant["file"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec["params"], f)
        params_path = f.name

    cfg = LlmConfig()
    cfg.base.model_class = ModelType(spec["model_class"])
    cfg.base.params = params_path
    # Pass the checkpoint explicitly: export_llama's own auto-download maps
    # model_class -> HUGGING_FACE_REPO_IDS, which points at the base weights
    # rather than the Instruct ones we publish.
    cfg.base.checkpoint = _checkpoint(spec["source"])

    # Runner constants come from the config -- the single source of truth.
    # export_llama derives the rest (vocab) from params.
    consts = variant.get("constants", {})
    cfg.base.metadata = json.dumps(
        {k: consts[k] for k in ("get_bos_id", "get_eos_ids") if k in consts}
    )
    # A real KV cache. Without one the graph re-runs the whole prefix for every
    # generated token: at the 2048 bound that is ~865 MB of activations per
    # step, which thermally reboots a Galaxy S26 Ultra rather than merely
    # running slowly.
    #
    # `use_sdpa_with_kv_cache` keeps the cache as an in-place custom op, which
    # is also what Vulkan's `causal_sdpa` pattern matches on. Turning it off
    # instead leaves the cache as a FUNCTIONALIZED mutable buffer (cat +
    # slice_copy), and ExecuTorch then filters every spec on the output node as
    # a mutable buffer and trips `graph_output_allocated not set`
    # (exir/memory_planning.py:301).
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = True
    # Dynamic sequence length, so the runner prefills the whole prompt in one
    # call instead of one token at a time (ExecuTorch's TextPrefiller picks the
    # sequential path whenever `enable_dynamic_shape` is false). That is the
    # difference between ~600 and ~2200 prefill tok/s on an S26 Ultra.
    #
    # This needs the fix for ExecuTorch's memory planner landed alongside: with
    # dynamic shapes Vulkan partitions a subgraph whose only outputs are
    # symints, and the planner's verifier skips output nodes that carry no
    # tensor specs without recording that it did, then asserts
    # `graph_output_allocated not set` on a graph that is in fact fully planned.
    cfg.model.enable_dynamic_shape = True

    # Vulkan is 8da4w only: `get_vulkan_partitioner` asserts the dtype override
    # is fp32, so there is no bf16 counterpart to the XNNPACK variant.
    cfg.model.dtype_override = DtypeOverride.fp32
    # A recipe may name its own qmode (e.g. "torchao:8da8w" for int8 weights);
    # the rest of the dict are quantization knobs. An unset knob means
    # "executorch default" (see _params.py).
    knobs = dict(spec["quant"])
    cfg.quantization.qmode = knobs.pop("qmode", "8da4w")
    for knob, value in knobs.items():
        if value is not None:
            setattr(cfg.quantization, knob, value)

    cfg.backend.vulkan.enabled = True
    # Note that with dynamic shapes the exported token dimension is bounded at
    # `max_seq_length - 1` (extension/llm/export/builder.py) while the
    # `get_max_seq_len` method still reports `max_seq_length` -- the builder
    # overwrites that entry after the metadata override is applied, so it
    # cannot be corrected from here. The declared IO in config.json carries the
    # bound the graph really accepts (2047), matching the published XNNPACK
    # variant, which is exported through the same path and is off by the same
    # one token.
    cfg.export.max_seq_length = consts["get_max_seq_len"]
    cfg.export.max_context_length = consts["get_max_context_len"]
    cfg.export.output_name = out_path

    # Swap in our partitioner for the duration of the call: `export_llama`
    # builds its own and offers no hook for the compile options we need.
    from executorch.examples.models.llama import export_llama_lib

    original_partitioner = export_llama_lib.get_vulkan_partitioner
    export_llama_lib.get_vulkan_partitioner = _vulkan_partitioner

    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        export_llama(cfg)
    finally:
        export_llama_lib.get_vulkan_partitioner = original_partitioner
        os.chdir(cwd)
        os.unlink(params_path)
    return out_path
