"""Check exported depth contracts and parity on fixed-square mobile inputs.

uv run --project exporter_models/depth_anything_v2 python \
    exporter_models/depth_anything_v2/verify.py --backend xnnpack \
    --artifacts-dir /private/tmp/depth_anything_v2 --image /path/to/image.jpg

Use --inspect-only for Vulkan on a host without a Vulkan runtime.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import torch


def inspect_artifact(path, config, variant):
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(path.read_bytes()).program
    assert len(program.execution_plan) == 1
    plan = program.execution_plan[0]
    assert plan.name == 'forward'

    def describe(index):
        value = plan.values[index].val
        assert int(value.scalar_type) == 6, 'Expected float32 public IO'
        return {'shape': list(value.sizes), 'dtype': 'float32'}

    contract = {
        'inputs': [describe(i) for i in plan.inputs],
        'outputs': [describe(i) for i in plan.outputs],
    }
    assert contract == variant['methods']['forward'], contract
    expected_backend = {
        'xnnpack': 'XnnpackBackend', 'coreml': 'CoreMLBackend', 'vulkan': 'VulkanBackend',
    }[config['backend']]
    delegates = Counter(d.id for d in plan.delegates)
    assert delegates and set(delegates) == {expected_backend}, delegates
    instructions = Counter(
        type(i.instr_args).__name__ for chain in plan.chains for i in chain.instructions
    )
    report = {
        'contract': contract,
        'artifact_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'delegates': dict(delegates),
        'instructions': dict(instructions),
        'portable_operators': [o.name + '.' + o.overload for o in plan.operators],
    }
    if config['backend'] == 'vulkan':
        report['vulkan_storage'] = inspect_vulkan(program)
    return report


def inspect_vulkan(program):
    from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
        VkTensor, VkStorageType,
    )
    from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
        VulkanDelegateHeader, flatbuffer_to_vk_graph,
    )
    from executorch.backends.vulkan.utils import required_image_extents

    tensors = Counter()
    maximum = [0, 0, 0]
    for data in program.backend_delegate_data:
        payload = bytes(data.data)
        header = VulkanDelegateHeader.from_bytes(payload[:30])
        graph = flatbuffer_to_vk_graph(
            payload[header.flatbuffer_offset:header.flatbuffer_offset + header.flatbuffer_size]
        )
        for value in graph.values:
            value = value.value
            if not isinstance(value, VkTensor):
                continue
            tensors[value.storage_type.name] += 1
            if value.storage_type == VkStorageType.TEXTURE_3D:
                extents = required_image_extents(torch.Size(value.dims), value.memory_layout)
                assert all(v <= 2048 for v in extents), (value, extents)
                maximum = [max(a, b) for a, b in zip(maximum, extents)]
    return {'tensor_storage_counts': dict(tensors), 'maximum_3d_texture_extents': maximum}


def validate(path, image_path, backend, precision, threads, runs, trace_path):
    from PIL import Image
    from transformers import AutoImageProcessor
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch, _threadpool_get_thread_count, _unsafe_reset_threadpool,
    )
    from _model import SOURCE, REVISION, build_model

    torch.set_num_threads(threads)
    _unsafe_reset_threadpool(threads)
    reference = build_model()
    processor = AutoImageProcessor.from_pretrained(SOURCE, revision=REVISION, backend='pil')
    runtime = _load_for_executorch(str(path), enable_etdump=trace_path is not None)
    metadata = runtime.method_meta('forward')
    assert list(metadata.input_tensor_meta(0).sizes()) == [1, 3, 518, 518]
    assert list(metadata.output_tensor_meta(0).sizes()) == [1, 518, 518]
    cases = {}
    image_paths = sorted(image_path.glob('*.jpg')) if image_path.is_dir() else [image_path]
    if not image_paths:
        raise ValueError('No evaluation images found')
    for path_to_image in image_paths:
        with Image.open(path_to_image) as image:
            image = image.convert('RGB')
            variants = [(path_to_image.stem, image)] if image_path.is_dir() else [
                ('image', image), ('rotated', image.transpose(Image.Transpose.ROTATE_90)),
            ]
            for name, img in variants:
                # Deliberate fixed-square mobile policy, not upstream variable aspect resizing.
                cases[name] = processor(
                    images=img, return_tensors='pt', keep_aspect_ratio=False,
                )['pixel_values'].contiguous()
    cases['random'] = torch.randn(1, 3, 518, 518, generator=torch.Generator().manual_seed(42))
    cases['zeros'] = torch.zeros(1, 3, 518, 518)
    results = []
    with torch.inference_mode():
        for name, x in cases.items():
            expected = reference(x).clone()
            start = time.perf_counter()
            actual = runtime.forward([x])[0].clone()
            elapsed = (time.perf_counter() - start) * 1000
            assert actual.dtype == torch.float32 and actual.shape == expected.shape
            assert torch.isfinite(actual).all()
            error = (actual - expected).abs()
            rmse = error.square().mean().sqrt().item()
            span = (expected.max() - expected.min()).item()
            correlation = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
            # Quantized/FP16 screening checks structure and scale, not dataset depth accuracy.
            relative_rmse_limit = 0.01 if precision == 'fp16' else 0.02
            passed = (
                torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
                if precision == 'fp32' else rmse / max(span, 1e-6) < relative_rmse_limit and correlation > 0.99
            )
            result = {
                'case': name, 'mae': error.mean().item(), 'max_abs_error': error.max().item(),
                'rmse': rmse, 'rmse_over_reference_range': rmse / max(span, 1e-6),
                'correlation': correlation, 'min': actual.min().item(), 'max': actual.max().item(),
                'first_ms': elapsed, 'passed': passed,
            }
            print(json.dumps(result), flush=True)
            results.append(result)
        benchmark_input = next(iter(cases.values()))
        runtime.forward([benchmark_input])
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            runtime.forward([benchmark_input])
            timings.append((time.perf_counter() - start) * 1000)
    if trace_path is not None:
        runtime.write_etdump_result_to_file(str(trace_path))
        if not trace_path.is_file():
            raise RuntimeError('No trace written; use an ExecuTorch runtime built with event tracing')
    return {
        'cases': results, 'passed': all(r['passed'] for r in results),
        'torch_threads': torch.get_num_threads(), 'executorch_threads': _threadpool_get_thread_count(),
        'warm_latency_ms': {'samples': timings, 'median': statistics.median(timings)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['xnnpack', 'coreml', 'vulkan'], required=True)
    parser.add_argument('--variant', default=None, help='Precision name; defaults to the configured default')
    parser.add_argument('--artifacts-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--trace', type=Path)
    args = parser.parse_args()
    if not args.inspect_only and args.image is None:
        parser.error('--image is required for runtime validation')
    if args.threads < 1 or args.runs < 1:
        parser.error('--threads and --runs must be positive')
    config = json.loads((Path(__file__).parent / 'small' / args.backend / 'config.json').read_text())
    variants = [v for v in config['variants'] if v['precision'] == args.variant] if args.variant else [
        v for v in config['variants'] if v['default']
    ]
    if len(variants) != 1:
        parser.error('Select exactly one configured precision with --variant')
    variant = variants[0]
    path = args.artifacts_dir / args.backend / variant['file']
    report = inspect_artifact(path, config, variant)
    report.update({
        'backend': args.backend, 'platform': platform.platform(),
        'precision': variant['precision'],
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'executorch', 'transformers']},
        'runtime_tested': not args.inspect_only,
    })
    if not args.inspect_only:
        report.update(validate(path, args.image, args.backend, variant['precision'], args.threads, args.runs, args.trace))
    suffix = f'_{args.variant}' if args.variant else ''
    destination = args.artifacts_dir / f'{args.backend}{suffix}_verification.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report.get('passed') is False:
        raise SystemExit('Numerical validation failed; see ' + str(destination))


if __name__ == '__main__':
    main()
