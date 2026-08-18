# Supertonic 3 ExecuTorch Export

Supertonic 3 is an ultra-lightweight, high-quality multilingual Text-to-Speech (TTS) pipeline based on a VITS-style text encoder, duration predictor, CFG flow-matching ODE vector estimator, and neural vocoder.

Exporting emits four `.pte` sub-models (`multi_pte`) mirroring the ONNX layout:

1. `duration_predictor` — predicts frame duration per text token.
2. `text_encoder` — encodes text tokens + voice style conditioning.
3. `vector_estimator` — single Euler step of the CFG flow-matching ODE.
4. `vocoder` — decodes latent sequence into output audio waveform.

---

## Requirements & Custom Build Notice

> [!IMPORTANT]
> **Custom ExecuTorch Runtime Build Required for MLX**
> 
> Running the MLX models (`*_mlx_fp32.pte`) on device in [`react-native-executorch`](https://github.com/software-mansion/react-native-executorch) requires native C++ binaries built from Software Mansion's ExecuTorch fork:
> 
> **Repository:** [`software-mansion-labs/executorch`](https://github.com/software-mansion-labs/executorch)
> 
> ### Why the Custom Build is Required:
> 
> 1. **Graph Partitioning & Pattern Passes**:
>    Software Mansion's custom MLX backend exporter ([`backends/mlx`](https://github.com/software-mansion-labs/executorch/tree/main/backends/mlx)) applies custom source graph transformations (such as select-as-view and RMSNorm decomposition) before partitioning, producing the clean **2-subgraph structure** expected by the MLX interpreter.
> 
> 2. **Runtime GPU Memory Release (`Error::Internal` Fix)**:
>    Supertonic runs `vector_estimator` in an **8-step flow-matching ODE loop**. In standard ExecuTorch, intermediate Metal GPU memory is retained until `execute()` finishes, accumulating > 2 GB of peak memory. Software Mansion's C++ MLX runtime ([commit `5133162`](https://github.com/software-mansion-labs/executorch/commit/513316292e8de51947236f1f5cbe83cbd3c63182)) drops temporary tensor slots at last use during graph evaluation, bounding GPU memory usage and preventing native `Error::Internal` crashes.

---

## Dynamic Shapes & Companion Schema

Each exported `.pte` carries a companion constant method named `get_model_schema` returning a compact `ModelSpec<ConcreteDim>` JSON string keyed by `"forward"`:

```json
{
  "forward": {
    "inputs": [...],
    "outputs": [...],
    "runtimeConstraints": [...]
  }
}
```

### Bounds:
- **Text Length (`T`)**: `min: 8, max: 256, step: 1`
  - *Note:* `MAX_TEXT = 256` bounds the quadratic text attention activation memory in ExecuTorch's memory planner.
- **Latent Length (`L`)**: `min: 4, max: 1000, step: 1`

---

## How to Run Export

From the root of `export-scripts`:

### XNNPACK (CPU)
```bash
uv run exporter export supertonic-3 --backend xnnpack --output-dir checkpoints/xnnpack
```

### MLX (Apple Silicon GPU)
```bash
uv run exporter export supertonic-3 --backend mlx --output-dir checkpoints/mlx
```

The MLX export script automatically verifies and syncs Software Mansion's MLX exporter passes if a standard PyPI `executorch` installation is active.
