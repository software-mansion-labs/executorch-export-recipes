"""Export full FP32 Fluid Speech Pro as independently loadable components."""
import importlib.util
from pathlib import Path
import sys
import gc
import torch

ROOT = Path(__file__).resolve().parent.parent


def export(config: dict, out_dir: str) -> None:
    from torch.export import Dim, export as capture
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig, ExecutorchBackendConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from nemo.core.classes import typecheck

    spec = importlib.util.spec_from_file_location('fluid_speech_export_model', ROOT / '_model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    model = module.prepare_capture(module.load_source())
    # Precompute positional encodings before graph capture; no source weights change.
    model.encoder.set_max_audio_length(1001)
    zero = torch.zeros
    variants = config.get('variants', [config.get('variant')])
    selected = {v['file'] for v in variants if v}
    boundaries = [
        ('preprocess', module.Features(model).eval(), (zero(1, 160000), torch.tensor([160000])), None),
        ('predictor', module.Predict(model).eval(), (torch.tensor([[8192]]), zero(2, 1, 640), zero(2, 1, 640)), None),
        ('joint', module.Joint(model).eval(), (zero(1, 1, 1024), zero(1, 1, 640)), None),
        ('subsample', module.Subsample(model).eval(), (zero(1, 128, 501), torch.tensor([500])),
         ({2: Dim('mel_frames', min=17, max=1001)}, {})),
        ('encoder', module.Transformer(model).eval(), (zero(1, 63, 1024), torch.tensor([63])),
         ({1: Dim('encoded_frames', min=3, max=126)}, {})),
    ]
    with torch.inference_mode(), typecheck.disable_checks():
        for name, wrapper, inputs, dynamic in boundaries:
            filename = f'fluid_speech_pro_{name}_xnnpack_fp32.pte'
            if filename not in selected:
                continue
            print(f'Capturing {name}', flush=True)
            if name == 'subsample':
                # Cover all eight time-axis residues exactly. Derived stride axes
                # avoid Torch symbolic floor/divisibility guards without padding.
                ep = {}
                for residue in range(8):
                    low = (17-residue+7)//8
                    high = (1001-residue)//8
                    groups = Dim(f'groups_{residue}', min=low, max=high)
                    example = 8*62+residue
                    ep[f'forward_{residue}'] = capture(wrapper, (zero(1,128,example),torch.tensor([example-1])),
                        dynamic_shapes=({2:8*groups+residue},{}),strict=False)
            else:
                ep = capture(wrapper, inputs, dynamic_shapes=dynamic, strict=False)
            print(f'Lowering {name}', flush=True)
            edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False))
            program = edge.to_executorch(config=ExecutorchBackendConfig())
            path = Path(out_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as f:
                program.write_to_file(f)
            print(f'Wrote {path}: {path.stat().st_size} bytes', flush=True)
            del ep, edge, program
            gc.collect()
