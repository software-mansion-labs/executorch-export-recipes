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

To run the export script, use the following command:

```bash
uv run python export_fastsam.py
```

To run the demo script, use the following command:

```bash
uv run python fastsam_demo.py
```
