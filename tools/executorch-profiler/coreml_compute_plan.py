#!/usr/bin/env python3
"""
Report which CoreML operations dispatch to ANE / GPU / CPU.

This is the CoreML counterpart to the XNNPACK timing/delegation view produced
by profile_et.py. The CoreML runtime decides at compile/load time which compute
device (Apple Neural Engine / GPU / CPU) each delegated op runs on; that
decision is exposed by MLComputePlan in coremltools 9.0+, so we can answer
"why isn't my model running on the ANE?" without writing any Swift.

The script accepts a CoreML-backed ExecuTorch .pte (every CoreML partition is
extracted and analyzed) or a CoreML model directly (.mlpackage / .mlmodelc).

By default the plan uses the compute_units the model was actually compiled with
(read from the .pte's CoreML compile spec). The plan reflects the current host
and can differ on device; pass --compute-units to explore a different setting. Note the ANE is
fp16-only, so a float32 model will never dispatch to the ANE regardless.

Usage:
    python coreml_compute_plan.py --model=exported_model.pte
    python coreml_compute_plan.py --model=model.mlpackage --show-non-ane
    python coreml_compute_plan.py --model=exported_model.pte --show-ops
    python coreml_compute_plan.py --model=exported_model.pte --compute-units=all

Requires coremltools >= 9.0 and an ExecuTorch built with CoreML enabled
(see build_et.sh --coreml).

Adapted from executorch examples/apple/coreml/scripts (pytorch/executorch#19252),
with the .pte extraction vendored inline so the tool stays self-contained.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import coremltools as ct
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import MLComputePlan
from colorama import Fore, Style, init as colorama_init


# --------------------------------------------------------------------------- #
# .pte extraction (vendored from executorch extract_coreml_models.py so this   #
# tool does not depend on the exact installed executorch signature).           #
# --------------------------------------------------------------------------- #

COREML_BACKEND_ID = "CoreMLBackend"


def compute_units_from_pte(pte_data: bytes) -> Optional[str]:
    """Return the ``compute_units`` the CoreML partitions were compiled with.

    The executorch CoreML backend bakes ``compute_units`` (e.g. ``all``,
    ``cpu_and_ne``) into each delegate's compile specs. That is what the runtime
    actually uses on device, so it makes a better default for the compute plan
    than a fixed guess. Returns None if no CoreML delegate declares it, and the
    first value (with a warning) if partitions disagree.
    """
    from executorch.exir._serialize._program import deserialize_pte_binary

    try:
        program = deserialize_pte_binary(pte_data).program
    except Exception:
        return None

    values: List[str] = []
    for execution_plan in program.execution_plan:
        for delegate in execution_plan.delegates:
            if delegate.id != COREML_BACKEND_ID:
                continue
            for spec in delegate.compile_specs:
                if spec.key == "compute_units":
                    value = bytes(spec.value).decode("utf-8", "ignore").strip()
                    if value:
                        values.append(value)

    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) > 1:
        print(
            f"Warning: CoreML partitions declare different compute_units "
            f"({', '.join(unique)}); using '{unique[0]}'.",
            file=sys.stderr,
        )
    return unique[0]


def extract_coreml_models(
    pte_data: bytes,
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Extract every CoreML delegate partition from a .pte into ``out_dir``."""
    from executorch.backends.apple.coreml import executorchcoreml
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import (
        BackendDelegate,
        BackendDelegateDataReference,
        DataLocation,
    )

    out_root = Path(out_dir) if out_dir is not None else Path("extracted_coreml_models")

    pte_file = deserialize_pte_binary(pte_data)
    program = pte_file.program

    # Multifunction models store a JSON reference in processed_bytes that points
    # to the actual model data in named_data; build that lookup up front.
    named_data_map: Dict[str, bytes] = {}
    if pte_file.named_data is not None:
        for key, data_entry in pte_file.named_data.pte_data.items():
            named_data_map[key] = pte_file.named_data.buffers[data_entry.buffer_index]

    delegates: List[BackendDelegate] = sum(
        [execution_plan.delegates for execution_plan in program.execution_plan], []
    )
    coreml_delegates: List[BackendDelegate] = [
        delegate for delegate in delegates if delegate.id == COREML_BACKEND_ID
    ]

    extracted_paths: List[Path] = []
    extracted_keys: set = set()
    model_index: int = 1

    for coreml_delegate in coreml_delegates:
        coreml_delegate_data: BackendDelegateDataReference = coreml_delegate.processed
        coreml_processed_bytes: Optional[bytes] = None
        model_name: Optional[str] = None

        if coreml_delegate_data.location == DataLocation.INLINE:
            raw_bytes = program.backend_delegate_data[coreml_delegate_data.index].data

            # JSON references to named_data (multifunction) are prefixed "CMJR".
            MAGIC_NUMBER = b"CMJR"
            if raw_bytes.startswith(MAGIC_NUMBER):
                json_bytes = raw_bytes[len(MAGIC_NUMBER):]
                try:
                    reference = json.loads(json_bytes.decode("utf-8"))
                    key = reference.get("key")
                    if key in extracted_keys:
                        continue
                    if key in named_data_map:
                        coreml_processed_bytes = named_data_map[key]
                        model_name = key
                        extracted_keys.add(key)
                    else:
                        print(f"Warning: Named data key '{key}' not found in program")
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Warning: Failed to parse JSON reference: {e}")
                    continue
            else:
                coreml_processed_bytes = raw_bytes
        else:
            raise AssertionError("The loaded Program must have inline data.")

        if coreml_processed_bytes is None:
            continue

        if model_name is None:
            model_name = f"model_{model_index}"

        model_path: Path = out_root / model_name
        if model_path.exists():
            shutil.rmtree(model_path.absolute())
        os.makedirs(model_path.absolute())

        if executorchcoreml.unflatten_directory_contents(
            coreml_processed_bytes, str(model_path.absolute())
        ):
            print(f"Core ML model extracted to: {model_path}")
            extracted_paths.append(model_path)
        model_index += 1

    if len(coreml_delegates) == 0:
        print("The model isn't delegated to Core ML.")

    return extracted_paths


# --------------------------------------------------------------------------- #
# Compute-plan analysis.                                                        #
# --------------------------------------------------------------------------- #

_DEVICE_NAMES: List[Tuple[type, str]] = [
    (MLNeuralEngineComputeDevice, "ANE"),
    (MLGPUComputeDevice, "GPU"),
    (MLCPUComputeDevice, "CPU"),
]

_COMPUTE_UNIT_CHOICES = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}

# ANE is the goal; GPU is acceptable; CPU is the fallback you usually don't want.
_DEVICE_COLORS = {
    "ANE": Fore.GREEN + Style.BRIGHT,
    "GPU": Fore.YELLOW + Style.BRIGHT,
    "CPU": Fore.RED + Style.BRIGHT,
    "unknown": Style.BRIGHT,
}


def _device_name(device) -> str:
    if device is None:
        return "unknown"
    for cls, name in _DEVICE_NAMES:
        if isinstance(device, cls):
            return name
    return type(device).__name__


def _iter_operations(block) -> Iterable:
    for op in block.operations:
        yield op
        for nested in getattr(op, "blocks", None) or []:
            yield from _iter_operations(nested)


def _ensure_compiled(model_path: str, tmpdir: str) -> str:
    """Return a ``.mlmodelc`` path; compile from ``.mlpackage`` if needed."""
    if model_path.endswith(".mlmodelc"):
        return model_path
    if model_path.endswith(".mlpackage"):
        dest = os.path.join(
            tmpdir, os.path.basename(model_path).replace(".mlpackage", ".mlmodelc")
        )
        return str(ct.models.utils.compile_model(model_path, destination_path=dest))
    raise ValueError(f"Expected a .mlpackage or .mlmodelc path, got: {model_path}")


def _find_coreml_model(root: str) -> str:
    """Resolve an extracted partition dir to the .mlpackage/.mlmodelc inside it.

    ``extract_coreml_models`` returns the partition root; depending on the
    executorch version the actual model is either that dir itself or nested
    under ``lowered_module/`` (e.g. ``<root>/lowered_module/model.mlpackage``).
    """
    if root.endswith((".mlpackage", ".mlmodelc")):
        return root
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.endswith((".mlpackage", ".mlmodelc")):
                return os.path.join(dirpath, d)
    raise FileNotFoundError(
        f"No .mlpackage or .mlmodelc found under extracted partition: {root}"
    )


def _mlpackage_function_names(model_path: str) -> List[str]:
    """Names of the MLProgram functions inside an .mlpackage, or [] otherwise."""
    if not model_path.endswith(".mlpackage"):
        return []
    spec = ct.models.MLModel(model_path, skip_model_load=True).get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        return []
    return list(spec.mlProgram.functions.keys())


def _project_to_single(src_mlpackage: str, function_name: str, tmpdir: str) -> str:
    """Re-save ``src_mlpackage`` with only ``function_name`` exposed as ``main``."""
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    dest = os.path.join(tmpdir, f"{function_name}.mlpackage")
    desc = MultiFunctionDescriptor()
    desc.add_function(
        src_mlpackage,
        src_function_name=function_name,
        target_function_name="main",
    )
    desc.default_function_name = "main"
    save_multifunction(desc, dest)
    return dest


def _analyze_compiled(
    model_path: str, compute_units: ct.ComputeUnit
) -> List[Tuple[str, str, str]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        compiled = _ensure_compiled(model_path, tmpdir)
        plan = MLComputePlan.load_from_path(compiled, compute_units=compute_units)
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError(
                f"{model_path} is not an MLProgram model; this tool only supports "
                "the MLProgram backend (the CoreML backend executorch produces today)."
            )

        rows: List[Tuple[str, str, str]] = []
        for fname, fn in program.functions.items():
            for op in _iter_operations(fn.block):
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    # Constants and similar non-dispatched ops don't have a plan.
                    continue
                rows.append(
                    (
                        fname,
                        op.operator_name,
                        _device_name(usage.preferred_compute_device),
                    )
                )
        return rows


def analyze_one(
    model_path: str, compute_units: ct.ComputeUnit
) -> List[Tuple[str, str, str]]:
    """Return [(function, operator_name, device)] for every op that has a plan.

    coremltools 9.0's ``MLComputePlan.load_from_path`` only exposes usage for
    the default function of a multifunction package, so a multifunction
    .mlpackage is analyzed function-by-function by projecting each function as
    the ``main`` of a temp single-function copy.
    """
    function_names = _mlpackage_function_names(model_path)
    if len(function_names) <= 1:
        return _analyze_compiled(model_path, compute_units)
    rows: List[Tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in function_names:
            projected = _project_to_single(model_path, fname, tmpdir)
            for _, op_name, device in _analyze_compiled(projected, compute_units):
                rows.append((fname, op_name, device))
    return rows


def _print_op_breakdown(title: str, color: str, ops: Counter) -> None:
    if not ops:
        return
    total = sum(ops.values())
    print(color + f"\n  {title} ({total}):" + Style.RESET_ALL)
    for op_name, count in ops.most_common():
        print(f"    {count:5d}  {op_name}")


def _print_report(
    label: str,
    rows: List[Tuple[str, str, str]],
    show_non_ane: bool,
    show_ops: bool,
) -> None:
    print(Fore.BLUE + Style.BRIGHT + f"\n=== {label} ===" + Style.RESET_ALL)
    if not rows:
        print("  (no dispatched operations found)")
        return
    by_device = Counter(device for _, _, device in rows)
    total = sum(by_device.values())
    for device in ("ANE", "GPU", "CPU", "unknown"):
        count = by_device.get(device, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total
        color = _DEVICE_COLORS.get(device, "")
        print(color + f"  {device}: {count:5d} / {total} ({pct:5.1f}%)" + Style.RESET_ALL)

    if show_ops:
        # Per-device op-type breakdown: the ANE-delegated ops plus everything
        # that landed on the GPU / CPU instead.
        for device in ("ANE", "GPU", "CPU", "unknown"):
            ops = Counter(op_name for _, op_name, dev in rows if dev == device)
            _print_op_breakdown(
                f"{device} op types", _DEVICE_COLORS.get(device, ""), ops
            )
    elif show_non_ane:
        non_ane = Counter(op_name for _, op_name, dev in rows if dev != "ANE")
        _print_op_breakdown(
            "Non-ANE op types", Fore.YELLOW + Style.BRIGHT, non_ane
        )


def main() -> int:
    colorama_init(autoreset=True)

    parser = argparse.ArgumentParser(
        description="Report which CoreML ops dispatch to ANE / GPU / CPU."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a .pte, .mlpackage, or .mlmodelc.",
    )
    parser.add_argument(
        "--compute-units",
        default=None,
        choices=sorted(_COMPUTE_UNIT_CHOICES),
        help="Which devices the runtime may use when planning dispatch. "
        "Defaults to the compute_units the model was compiled with (read from "
        "the .pte), falling back to cpu_and_ne if that cannot be determined.",
    )
    parser.add_argument(
        "--show-non-ane",
        action="store_true",
        help="List op types that did not get assigned to the ANE.",
    )
    parser.add_argument(
        "--show-ops",
        action="store_true",
        help="List op types broken down per device (ANE / GPU / CPU), so both "
        "the delegated (ANE) ops and the rest are shown.",
    )
    args = parser.parse_args()

    model_path = args.model

    if model_path.endswith(".pte"):
        with open(model_path, "rb") as f:
            pte_data = f.read()
        baked = None if args.compute_units else compute_units_from_pte(pte_data)
        if args.compute_units is not None:
            cu_name, cu_source = args.compute_units, "override"
        elif baked is not None:
            cu_name, cu_source = baked, "model compile spec"
        else:
            cu_name, cu_source = "cpu_and_ne", "default"
        compute_units = _COMPUTE_UNIT_CHOICES[cu_name]
        print(
            Fore.CYAN
            + Style.BRIGHT
            + f"Compute units: {cu_name} ({cu_source})"
            + Style.RESET_ALL
        )
        with tempfile.TemporaryDirectory() as out_dir:
            extracted = extract_coreml_models(pte_data, out_dir=out_dir)
            if not extracted:
                print(
                    f"{model_path} does not contain any CoreML delegate partitions.",
                    file=sys.stderr,
                )
                return 1
            for path in extracted:
                model = _find_coreml_model(str(path))
                rows = analyze_one(model, compute_units)
                _print_report(path.name, rows, args.show_non_ane, args.show_ops)
    else:
        # A bare .mlpackage/.mlmodelc has no executorch compile spec to read.
        cu_name = args.compute_units or "cpu_and_ne"
        compute_units = _COMPUTE_UNIT_CHOICES[cu_name]
        source = "override" if args.compute_units else "default"
        print(
            Fore.CYAN + Style.BRIGHT + f"Compute units: {cu_name} ({source})" + Style.RESET_ALL
        )
        rows = analyze_one(model_path, compute_units)
        _print_report(
            os.path.basename(model_path.rstrip("/")),
            rows,
            args.show_non_ane,
            args.show_ops,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
