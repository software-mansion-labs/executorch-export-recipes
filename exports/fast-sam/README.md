# FastSAM export

This directory contains the exported
[FastSAM](https://docs.ultralytics.com/models/fast-sam/) model and a demo script
to run inference using the exported model.

## How to run

For reproducible results this repository uses [`uv`](https://docs.astral.sh/uv/)
to manage dependencies. To install dependencies run:

```bash
uv sync
```

To export to ExecuTorch XNNPACK format:

```bash
uv run python export_fastsam.py
uv run python export_fastsam.py --size s --input-size 640
uv run python export_fastsam.py --size x --output-dir /path/to/output
```

To export to CoreML format via ExecutorTorch:

```bash
uv run python export_fastsam_coreml.py
uv run python export_fastsam_coreml.py --size s --precision fp16
uv run python export_fastsam_coreml.py --size x --precision fp32 --input-size 1024
```

To run the demo script, use the following command:

```bash
uv run python fastsam_demo.py
```
