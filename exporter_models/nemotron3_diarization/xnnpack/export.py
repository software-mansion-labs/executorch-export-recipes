"""FP32 XNNPACK export with an explicit RNE dynamic schema and faithful cache compression."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent


def export(config: dict, out_dir: str) -> None:
    from torch.export import Dim, export as capture
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from exporter.schema import load_and_validate_schema

    spec = importlib.util.spec_from_file_location('nemotron3_diarization_export', ROOT / '_model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    model, processor = module.load_source()
    features = module.Features(processor.feature_extractor).eval()
    embed = module.Embed(model).eval()
    encode = module.Encode(model).eval()
    compress = module.Compress(model).eval()
    frame_groups = Dim('frame_groups', min=1, max=126)
    encoded_frames = Dim('encoded_frames', min=2, max=800)
    cache_frames = Dim('cache_frames', min=265, max=800)
    with torch.no_grad():
        print('Capturing preprocessing', flush=True)
        methods = {'preprocess': capture(features, (torch.zeros(160000), torch.tensor([160000])), strict=True)}
        print('Capturing embedding', flush=True)
        methods['embed'] = capture(embed, (torch.zeros(1, 1008, 128),), dynamic_shapes=({1: 8 * frame_groups},), strict=True)
        print('Capturing encoder', flush=True)
        methods['encode'] = capture(encode, (torch.zeros(1, 126, 512), torch.ones(1, 126, dtype=torch.int64)),
                                    dynamic_shapes=({1: encoded_frames}, {1: encoded_frames}), strict=True)
        print('Capturing cache compression', flush=True)
        methods['compress'] = capture(compress, (torch.zeros(1, 400, 512), torch.full((1, 400, 8), .1)),
                                      dynamic_shapes=({1: cache_frames}, {1: cache_frames}), strict=True)
        print('Lowering to XNNPACK', flush=True)
        constants = dict(config['variant'].get('constants', {}))
        constants['get_model_schema'] = load_and_validate_schema(ROOT / 'schema.json')
        program = to_edge_transform_and_lower(methods, partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False), constant_methods=constants)
        result = program.to_executorch(config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=True)))
        path = Path(out_dir) / config['variant']['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.buffer)
        print(f'Wrote {path} ({path.stat().st_size} bytes)', flush=True)
