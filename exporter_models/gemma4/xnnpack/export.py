import glob
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

logger = logging.getLogger(__name__)


def _load_text_decoder():
    pkg_init = _MODEL_DIR / "_text_decoder" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_text_decoder",
        pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_text_decoder")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_text_decoder"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_checkpoint(arg: str | None = None) -> str:
    if arg and os.path.exists(arg):
        return os.path.abspath(arg)
    candidates = [
        "/tmp/gemma4_hf",
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*"),
    ]
    for c in candidates:
        if "*" in c:
            hits = glob.glob(os.path.join(c, "model.safetensors"))
            if hits:
                return os.path.dirname(hits[0])
        else:
            if os.path.exists(os.path.join(c, "model.safetensors")):
                return os.path.abspath(c)
    raise SystemExit("Gemma 4 checkpoint not found with model.safetensors. Pass checkpoint_path explicitly.")


def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )
    from executorch.examples.models.gemma4.quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
    )
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.pass_base import ExportPass
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim

    td = _load_text_decoder()
    Gemma4Config = td.Gemma4Config
    Gemma4Model = td.Gemma4Model

    variant = config["variant"]
    consts = variant.get("constants", {})
    max_ctx = consts.get("get_max_context_len", 4048)
    max_seq = consts.get("get_max_seq_len", 4048)
    vocab_sz = consts.get("get_vocab_size", 262144)

    out_dir = Path(out_dir).resolve()
    out_path = str(out_dir / variant["file"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(None)

    gemma4_config = Gemma4Config.from_config("e2b")
    gemma4_config.use_kv_cache = True
    gemma4_config.max_seq_len = max_ctx
    gemma4_config.enable_dynamic_shape = True
    gemma4_config.use_custom_sdpa = False

    logger.info("Building Gemma4 E2B eager text model (fp32)...")
    model_wrapper = Gemma4Model(
        config=gemma4_config,
        checkpoint_path=checkpoint_path,
        dtype=torch.float32,
    )
    model = model_wrapper.get_eager_model()
    model.eval()

    logger.info("Quantizing linear layers to 8da4w (group_size 32)...")
    model = apply_linear_quantization(model, "8da4w", group_size=32)
    model.eval()

    # Dynamic inputs for text forward:
    # tokens: [1, seq_len] in [1, max_seq - 1]
    # input_pos: [1] (downstream runner passes rank-1 scalar [curPos])
    seq_dim = Dim("seq_len", min=1, max=max_seq - 1)
    tokens = torch.ones((1, 16), dtype=torch.long)
    input_pos = torch.zeros((1,), dtype=torch.long)
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "input_pos": {},
    }

    logger.info("Tracing model with torch.export...")
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (tokens,),
                kwargs={"input_pos": input_pos},
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )

    ep = ep.run_decompositions({})

    # Build companion schema conforming to model.schema.json
    model_schema = {
        "forward": {
            "inputs": [
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {
                            "kind": "range",
                            "range": {
                                "min": 1,
                                "max": max_seq - 1,
                                "step": 1,
                            },
                        },
                    ],
                },
                {
                    "kind": "Tensor",
                    "dtype": "int64",
                    "shape": [
                        {"kind": "constant", "value": 1}
                    ],
                },
            ],
            "outputs": [
                {
                    "kind": "Tensor",
                    "dtype": "float32",
                    "shape": [
                        {"kind": "constant", "value": 1},
                        {"kind": "constant", "value": vocab_sz},
                    ],
                }
            ],
            "runtimeConstraints": [],
        }
    }

    root_schema_path = _MODEL_DIR.parent / "model.schema.json"
    if root_schema_path.exists():
        import jsonschema

        root_schema = json.loads(root_schema_path.read_text())
        jsonschema.validate(instance=model_schema, schema=root_schema)

    runner_constants = {
        "enable_dynamic_shape": True,
        "get_bos_id": 2,
        "get_eos_ids": [1, 106, 50],
        "get_max_context_len": max_ctx,
        "get_max_seq_len": max_seq,
        "get_vocab_size": vocab_sz,
        "use_kv_cache": True,
        "get_model_schema": json.dumps(model_schema),
    }

    # Transform passes: bitwise lowering
    class _ReplaceBitwiseScalarPass(ExportPass):
        def __init__(self, ops_map):
            self.ops_map = ops_map
            super().__init__()

        def call_operator(self, op, args, kwargs, meta):
            if op not in self.ops_map:
                return super().call_operator(op, args, kwargs, meta)
            full_op = exir_ops.edge.aten.full.default
            scalar_as_tensor = super().call_operator(
                full_op,
                ((1,), float(args[1])),
                {
                    "dtype": args[0].to_tensor().dtype,
                    "device": args[0].to_tensor().device,
                },
                meta,
            )
            return super().call_operator(
                self.ops_map[op], (args[0], scalar_as_tensor, *args[2:]), kwargs, meta
            )

    bitwise_ops = {
        exir_ops.edge.aten.__rshift__.Scalar: exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.__lshift__.Scalar: exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.__and__.Scalar: exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.__or__.Scalar: exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.__xor__.Scalar: exir_ops.edge.aten.bitwise_xor.Tensor,
    }

    logger.info("Lowering to XNNPACK backend...")
    edge_program = exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=[_ReplaceBitwiseScalarPass(bitwise_ops)],
        partitioner=[
            XnnpackDynamicallyQuantizedPartitioner(),
            XnnpackPartitioner(),
        ],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=runner_constants,
    )

    logger.info("Serializing to ExecuTorch binary...")
    et_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    with open(out_path, "wb") as f:
        f.write(et_program.buffer)

    logger.info(f"Saved {out_path} ({len(et_program.buffer) / 1024 / 1024:.1f} MB)")
    return out_path
