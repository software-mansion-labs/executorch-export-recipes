"""Export Fluid Speech Pro components, with optional experimental INT8 linears."""
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
    schema_spec = importlib.util.spec_from_file_location('fluid_speech_schema', ROOT / '_schema.py')
    schema_module = importlib.util.module_from_spec(schema_spec)
    schema_spec.loader.exec_module(schema_module)
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
            quantized = name == 'encoder' and config.get('quantize_encoder') == 'int8dyn'
            precision = 'int8dyn' if quantized else 'fp32'
            filename = f'fluid_speech_pro_{name}_xnnpack_{precision}.pte'
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
            if quantized:
                from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
                from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
                quantizer = XNNPACKQuantizer().set_operator_type(
                    torch.ops.aten.linear.default,
                    get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
                print('Quantizing encoder linears: dynamic INT8 activations / per-channel INT8 weights', flush=True)
                prepared = prepare_pt2e(ep.module(), quantizer)
                prepared(*inputs)  # Initializes weight observers; no speech calibration.
                converted = convert_pt2e(prepared)
                ep = capture(converted, inputs, dynamic_shapes=dynamic, strict=False)
                # Backend parity fixtures are saved before lowering, independent
                # of the original FP32/source task-quality comparison.
                torch.manual_seed(17)
                checks = []
                for frames in [3, 63, 126]:
                    values = (torch.randn(1, frames, 1024), torch.tensor([frames-1]))
                    checks.append((values, tuple(x.clone() for x in converted(*values))))
                torch.save(checks, Path(out_dir) / 'encoder-quantized-reference.pt')
                del prepared, converted
            print(f'Lowering {name}', flush=True)
            partitioners = [XnnpackPartitioner()]
            if quantized:
                from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
                # Separate quantized islands before merging float regions: the
                # combined greedy partitioner can create dependency cycles.
                partitioners = [XnnpackPartitioner(config_precisions=ConfigPrecisionType.DYNAMIC_QUANT, per_op_mode=True),
                                XnnpackPartitioner(config_precisions=ConfigPrecisionType.FP32)]
                torch.export.save(ep, Path(out_dir) / 'encoder-quantized.pt2')
            edge = to_edge_transform_and_lower(ep, partitioner=partitioners,
                constant_methods=schema_module.dynamic_schema(name),
                compile_config=EdgeCompileConfig(_check_ir_validity=False))
            program = edge.to_executorch(config=ExecutorchBackendConfig())
            path = Path(out_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as f:
                program.write_to_file(f)
            print(f'Wrote {path}: {path.stat().st_size} bytes', flush=True)
            del ep, edge, program
            gc.collect()
