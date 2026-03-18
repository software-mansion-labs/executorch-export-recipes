"""
Sweep quantization configs for an ExecuTorch LLM, benchmark each variant
(size, throughput, perplexity), and produce a comparison CSV + HTML plot.

Prerequisites
-------------
The ExecuTorch source tree pointed to by --executorch-dir must have:
  - A .venv virtualenv at <executorch-dir>/.venv with ExecuTorch installed
    from source (run the provided install_executorch.sh / install_requirements.sh
    scripts inside that repo before using this tool).

Path notes
----------
--config-yaml and --model-params may be given as paths relative to
--executorch-dir, or as absolute paths.
"""

import os
import time
import shutil
import argparse
import subprocess

from typing import Any
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd

import torch
import torch.nn.functional as F

from tqdm import trange, tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from optimum.executorch import ExecuTorchModelForCausalLM

from plot_utils import create_plot


class ETModelForCausalLMWrapper:
    """
    Thin wrapper around `optimum.executorch.ExecuTorchModelForCausalLM` that
    bundles the tokenizer, exposes model metadata (KV cache capacity, file
    size), and provides named constructors for local and HuggingFace Hub
    loading.
    """

    def __init__(self, *, _pte_path: Path, _tokenizer_repo: str):
        """
        Should not be called directly; use `from_hf` or `from_local`
        constructors.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_repo)
        self._pte_path = _pte_path
        self._model = ExecuTorchModelForCausalLM.from_pretrained(str(_pte_path.parent))

    @property
    def kv_cache_capacity(self) -> int:
        """Total positions the KV cache can hold."""
        return self._model.model.run_method("get_max_context_len")[0]  # type: ignore

    @property
    def size_mb(self) -> float:
        return self._pte_path.stat().st_size / 1024 / 1024

    @classmethod
    def from_hf(cls, repo: str, model_path: str) -> "ETModelForCausalLMWrapper":
        """Download .pte from HuggingFace Hub."""
        pte_path = Path(hf_hub_download(repo_id=repo, filename=model_path))
        return cls(_pte_path=pte_path, _tokenizer_repo=repo)

    @classmethod
    def from_local(cls, pte_path: Path, tokenizer_repo: str) -> "ETModelForCausalLMWrapper":
        """Load a locally exported .pte file."""
        return cls(_pte_path=pte_path, _tokenizer_repo=tokenizer_repo)

    def forward(self, input_ids: torch.Tensor, cache_pos: torch.Tensor) -> torch.Tensor:
        return self._model.forward(input_ids, cache_pos)  # type: ignore

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        return self._model.generate(input_ids, max_new_tokens)  # type: ignore


@dataclass
class ExportConfig:
    output_dir: Path
    executorch_dir: Path
    config_yaml: Path
    model_class: str
    model_params: Path
    max_seq_length: int
    max_context_length: int


def run_export(name: str, overrides: list[str], config: ExportConfig) -> Path:
    """
    Export a model to a .pte file using the ExecuTorch `export_llm` pipeline.

    Args:
       name (str):
            Identifier for this export (used as output filename and subdirectory).
        overrides (list[str]):
            Hydra overrides appended after the base config, e.g.
            ``["++quantization.mode=8da4w", "++quantization.group_size=32"]``.
            Use ``++key=value`` to upsert (add or replace) any config key.
        config (ExportConfig):
            Export configuration (model class, paths, sequence lengths, etc.).

    Returns:
        Path: path to the exported ``<name>.pte`` file.
    """
    cmd = [
        str(config.executorch_dir / ".venv/bin/python"),
        "-m",
        "extension.llm.export.export_llm",
        "--config",
        str(config.config_yaml),
        f"++base.model_class={config.model_class}",
        f"++base.params={config.model_params}",
        f"++export.max_seq_length={config.max_seq_length}",
        f"++export.max_context_length={config.max_context_length}",
        f"++export.output_name={name}",
        f"++export.output_dir={config.output_dir / name}",
        *overrides,
    ]
    print(f"\n[export] {name}")
    print("  " + " \\\n    ".join(cmd))

    # NOTE: Ensure output directory exists before running export, since the
    # 'export_llm' script doesn't create it and will error if it doesn't exist.
    (config.output_dir / name).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        cmd,
        cwd=config.executorch_dir,
        env={**os.environ, "PYTHONWARNINGS": "ignore::FutureWarning"},
    )

    if result.returncode != 0:
        raise RuntimeError(f"Export failed for '{name}' (exit {result.returncode})")

    if not (pte := config.output_dir / name / f"{name}.pte").exists():
        raise FileNotFoundError(f"Export claimed success but no .pte found at {pte}")

    return pte


class Benchmark(ABC):
    @abstractmethod
    def get_metrics_hib(self) -> dict[str, bool]:
        """
        Return ``{metric_name: higher_is_better}`` for each metric produced by
        ``run()``.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, Any]:
        """
        Run the benchmark on the given language model and return a dictionary of
        metric results.
        """
        raise NotImplementedError


class PerplexityBenchmark(Benchmark):
    def __init__(self, dataset: str, config: str, split: str, max_tokens: int):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.max_tokens = max_tokens

    def get_metrics_hib(self) -> dict[str, bool]:
        return {"ppl": False}

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        text = "\n\n".join(load_dataset(self.dataset, self.config, split=self.split)["text"])
        tokens = lm.tokenizer.encode(text)
        tokens = tokens[: 1 + min(lm.kv_cache_capacity, self.max_tokens)]
        tokens = torch.tensor([tokens], dtype=torch.long)

        nll = 0.0
        n = tokens.size(1) - 1

        for pos in trange(n):
            logits = lm.forward(tokens[:, pos : pos + 1], torch.tensor([pos], dtype=torch.long))
            target = tokens[:, pos + 1]
            nll += F.cross_entropy(logits, target).item()

        return {"ppl": float(torch.exp(torch.tensor(nll / n)))}


class SizeBenchmark(Benchmark):
    def get_metrics_hib(self) -> dict[str, bool]:
        return {"size_mb": False}

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        return {"size_mb": lm.size_mb}


class ThroughputBenchmark(Benchmark):
    def __init__(self, prompts: list[str], max_new_tokens: int, verbose: bool = False):
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    def get_metrics_hib(self) -> dict[str, bool]:
        return {"toks_per_sec": True}

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        thp = []
        for prompt in tqdm(self.prompts):
            inp = lm.tokenizer.encode(prompt)

            t = time.perf_counter()
            out = lm.generate(inp, self.max_new_tokens)
            t = time.perf_counter() - t

            thp.append((len(out) - len(inp)) / t)

            if self.verbose:
                print(lm.tokenizer.decode(out), "\n")

        return {"toks_per_sec": sum(thp) / len(thp)}


benchmarks: list[Benchmark] = [
    SizeBenchmark(),
    ThroughputBenchmark(
        prompts=[
            "In a distant galaxy far from our own, there existed",
            "The history of artificial intelligence began in the mid-twentieth century",
            "Once upon a time in a small village nestled between two mountains",
            "Scientists have recently discovered a new species of deep-sea creature",
            "The economic impact of climate change on developing nations",
        ],
        max_new_tokens=128,
    ),
    PerplexityBenchmark(
        dataset="wikitext",
        config="wikitext-2-raw-v1",
        split="test",
        max_tokens=512,
    ),
]


quantization_overrides: dict[str, list[str]] = {
    "fp32": [
        "++model.dtype_override=fp32",
    ],
    "bf16": [
        "++model.dtype_override=bf16",
    ],
    "8da4w_gs32": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
    ],
    "8da4w_gs64": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=64",
    ],
    "8da4w_gs128": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=128",
    ],
    "8da4w_gs32_emb8_0": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++quantization.embedding_quantize='8,0'",
    ],
    "8da4w_gs32_emb8_32": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++quantization.embedding_quantize='8,32'",
    ],
    "8da4w_gs32_emb8_64": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++quantization.embedding_quantize='8,64'",
    ],
    "8da4w_gs32_emb4_0": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++quantization.embedding_quantize='4,0'",
    ],
    "8da4w_gs64_emb8_0": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=64",
        "++quantization.embedding_quantize='8,0'",
    ],
    "8da4w_gs32_kvq": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++model.quantize_kv_cache=true",
    ],
    "8da4w_gs32_emb8_kvq": [
        "++quantization.qmode=8da4w",
        "++quantization.group_size=32",
        "++quantization.embedding_quantize='8,0'",
        "++model.quantize_kv_cache=true",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # fmt: off
    parser.add_argument("--executorch-dir", type=Path, required=True, help="Path to executorch source directory")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "exports", help="Directory to save .pte files and results")

    parser.add_argument("--config-yaml", type=Path, required=True, help="Path to base Hydra config YAML for export (absolute or relative to --executorch-dir)")
    parser.add_argument("--model-class", required=True, help="Model architecture identifier (e.g. 'lfm2_5_1_2b'). Must match a ModelType value in extension/llm/export/config/llm_config.py")
    parser.add_argument("--model-params", type=Path, required=True, help="Path to model params JSON file (absolute or relative to --executorch-dir)")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace repo for tokenizer")
    parser.add_argument("--max-seq-length", type=int, required=True, help="Max sequence length for export")
    parser.add_argument("--max-context-length", type=int, required=True, help="Max context length for export")

    parser.add_argument("--recompute-exports", action="store_true", help="Re-export even if .pte already exists")
    parser.add_argument("--recompute-benchmarks", action="store_true", help="Re-benchmark even if results already in results.csv")
    parser.add_argument("--discard-exports", action="store_true", help="Delete .pte files after benchmarking")
    # fmt: on

    args = parser.parse_args()

    plot_file: Path = args.output_dir / f"{args.model_class}_results.html"
    exports_dir: Path = args.output_dir / f"{args.model_class}"
    results_file: Path = args.output_dir / f"{args.model_class}_results.csv"

    config = ExportConfig(
        executorch_dir=args.executorch_dir,
        output_dir=exports_dir,
        config_yaml=args.config_yaml,
        model_class=args.model_class,
        model_params=args.model_params,
        max_seq_length=args.max_seq_length,
        max_context_length=args.max_context_length,
    )

    if args.recompute_benchmarks or not results_file.exists():
        df = pd.DataFrame(columns=["name"])
    else:
        df = pd.read_csv(results_file)

    for config_name, overrides in quantization_overrides.items():
        print(f"\n{'='*60}\n{config_name}\n{'='*60}")

        try:
            pte = exports_dir / config_name / f"{config_name}.pte"
            if args.recompute_exports or not pte.exists():
                pte = run_export(config_name, overrides, config)

            if args.recompute_benchmarks or config_name not in df["name"].values:
                metrics = {}
                for benchmark in benchmarks:
                    lm = ETModelForCausalLMWrapper.from_local(pte, args.tokenizer)
                    metrics |= benchmark.run(lm)

                df = pd.concat([df, pd.DataFrame([{"name": config_name, **metrics}])], ignore_index=True)
                print(df.to_string(index=False))

            if args.discard_exports:
                shutil.rmtree(pte.parent)

        except Exception as e:
            print(f"  [error] {e}")

    df.to_csv(results_file, index=False)

    metric_registry: dict[str, bool] = {}
    for benchmark in benchmarks:
        metric_registry |= benchmark.get_metrics_hib()

    create_plot(
        df,
        metric_registry=metric_registry,
        save_path=plot_file,
        title=f"{args.model_class} Benchmark Results",
    )


if __name__ == "__main__":
    main()
