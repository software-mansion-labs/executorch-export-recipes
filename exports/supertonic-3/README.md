# Supertonic 3 – Export Pipeline

```bash
# 1. Convert ONNX weights → PyTorch checkpoint
python -m supertonic_torch.convert --onnx-dir onnx --out checkpoints/supertonic3.pt

# 2. Export → ExecuTorch XNNPACK (.pte files)
python export-scripts/exports/supertonic-3/export_xnnpack.py \
    --checkpoint checkpoints/supertonic3.pt \
    --output-dir checkpoints/xnnpack

# 3. Verify PyTorch vs ONNX Runtime parity
python -m supertonic_torch.verify_parity --onnx-dir onnx

# 4. Benchmark XNNPACK (optionally verify against eager)
python export-scripts/exports/supertonic-3/benchmark_xnnpack.py \
    --model-pte-dir checkpoints/xnnpack
python export-scripts/exports/supertonic-3/benchmark_xnnpack.py \
    --model-pte-dir checkpoints/xnnpack --verify
```