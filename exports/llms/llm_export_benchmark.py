import os
import time
import shutil
import argparse
import subprocess

from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional, Self

import pandas as pd

import torch
import torch.nn.functional as F

from tqdm import trange, tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from optimum.executorch import ExecuTorchModelForCausalLM

from plot_utils import create_plot


os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")


class ETModelForCausalLMWrapper:
    def __init__(self, *, _pte_path: Path, _tokenizer_repo: str):
        """
        Should not be called directly; use `from_hf` or `from_local`
        constructors.
        """
        self._tokenizer_repo = _tokenizer_repo
        self._pte_path = _pte_path

        self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_repo)
        self._model: ExecuTorchModelForCausalLM
        self._model = ExecuTorchModelForCausalLM.from_pretrained(str(_pte_path.parent))  # type: ignore

    def reset(self) -> Self:
        """Reload the model from disk, clearing any cached state."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_repo)
        self._model = ExecuTorchModelForCausalLM.from_pretrained(str(self._pte_path.parent))  # type: ignore
        return self

    @property
    def kv_cache_capacity(self) -> int:
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

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        return self._model.forward(input_ids, cache_position)  # type: ignore

    def generate(self, prompt_tokens: list[int], max_seq_len: Optional[int] = None) -> list[int]:
        return self._model.generate(prompt_tokens, max_seq_len=max_seq_len)  # type: ignore

    def text_generate(self, prompt: str, max_seq_len: Optional[int] = None) -> str:
        return self._model.text_generation(
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_seq_len=max_seq_len,
        )  # type: ignore

    def copy(self) -> "ETModelForCausalLMWrapper":
        return type(self).from_local(pte_path=self._pte_path, tokenizer_repo=self._tokenizer_repo)


@dataclass
class ExportConfig:
    output_dir: Path
    executorch_dir: Path
    config_yaml: Path
    model_class: str
    model_params: Path
    max_seq_length: int
    max_context_length: int
    debug: bool
    checkpoint: Optional[Path] = None
    metadata: Optional[str] = None


def run_export(name: str, overrides: list[str], config: ExportConfig) -> Path:
    """
    Export a model to a .pte file using the ExecuTorch `export_llm` pipeline.

    Args:
        name (str):
           Identifier for this export (used as output filename and subdirectory).

        overrides (list[str]):
            Hydra overrides appended after the base config.

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
        f"++debug.verbose={True if config.debug else False}",
        *([f"++base.checkpoint={config.checkpoint}"] if config.checkpoint else []),
        *([f"++base.metadata='{config.metadata}'"] if config.metadata else []),
        *overrides,
    ]
    print(f"\n[export] {name}")
    print("  " + " \\\n    ".join(cmd))

    # NOTE: Ensure output directory exists before running export, since the
    # 'export_llm' script doesn't create it and will error if it doesn't exist.
    (config.output_dir / name).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd, cwd=config.executorch_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Export failed for '{name}' (exit {result.returncode})")

    if not (pte := config.output_dir / name / f"{name}.pte").exists():
        raise FileNotFoundError(f"Export claimed success but no .pte found at {pte}")

    return pte


class Benchmark(ABC):
    @abstractmethod
    def get_metrics_hib(self) -> dict[str, bool]:
        raise NotImplementedError

    @abstractmethod
    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, Any]:
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

        nll_total = 0.0
        n = tokens.size(1) - 1

        for pos in trange(n, desc="perplexity"):
            logits = lm.forward(tokens[:, pos : pos + 1], torch.tensor([pos], dtype=torch.long))
            target = tokens[:, pos + 1]
            nll_total += F.cross_entropy(logits, target).item()

        return {"ppl": float(torch.exp(torch.tensor(nll_total / n)))}


class SizeBenchmark(Benchmark):
    def get_metrics_hib(self) -> dict[str, bool]:
        return {"size_mb": False}

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        return {"size_mb": lm.size_mb}


class ThroughputBenchmark(Benchmark):
    def __init__(self, prompts: list[str], max_new_tokens: int):
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    def get_metrics_hib(self) -> dict[str, bool]:
        return {"toks_per_sec": True}

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        thp = []
        for prompt in tqdm(self.prompts, desc="throughput"):
            inp = lm.tokenizer.encode(prompt)
            t = time.perf_counter()
            out = lm.generate(inp, self.max_new_tokens)
            t = time.perf_counter() - t
            thp.append(len(out) / t)

        return {"toks_per_sec": sum(thp) / len(thp)}


class KLDivergenceBenchmark(Benchmark):
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        max_tokens: int,
        lmref: Optional[ETModelForCausalLMWrapper] = None,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.max_tokens = max_tokens
        self._lmref: Optional[ETModelForCausalLMWrapper] = lmref

    def get_metrics_hib(self) -> dict[str, bool]:
        return {"kl_div": False}

    def set_reference_model(self, lmref: ETModelForCausalLMWrapper) -> None:
        self._lmref = lmref

    def run(self, lm: ETModelForCausalLMWrapper) -> dict[str, float]:
        if self._lmref is None:
            raise RuntimeError("KLDivergenceBenchmark: call `set_reference_model()` before `run()`")
        self._lmref.reset()

        text = "\n\n".join(load_dataset(self.dataset, self.config, split=self.split)["text"])
        tokens = lm.tokenizer.encode(text)
        tokens = tokens[: 1 + min(lm.kv_cache_capacity, self.max_tokens)]
        tokens = torch.tensor([tokens], dtype=torch.long)

        kl_total = 0.0
        n = tokens.size(1) - 1

        for pos in trange(n, desc="kl_div"):
            p_logits = self._lmref.forward(tokens[:, pos : pos + 1], torch.tensor([pos], dtype=torch.long))
            q_logits = lm.forward(tokens[:, pos : pos + 1], torch.tensor([pos], dtype=torch.long))
            kl_total += F.kl_div(
                F.log_softmax(q_logits.float(), dim=-1),
                F.log_softmax(p_logits.float(), dim=-1),
                log_target=True,
                reduction="sum",
            ).item()

        return {"kl_div": kl_total / n}


PROMPTS = [
    "In a distant galaxy far from our own, there existed",
    "The history of artificial intelligence began in the mid-twentieth century",
    "Once upon a time in a small village nestled between two mountains",
    "Scientists have recently discovered a new species of deep-sea creature",
    "The economic impact of climate change on developing nations",
]
BENCHMARKS: list[Benchmark] = [
    SizeBenchmark(),
    ThroughputBenchmark(prompts=PROMPTS, max_new_tokens=128),
    PerplexityBenchmark(dataset="wikitext", config="wikitext-2-raw-v1", split="test", max_tokens=1000),
    KLDivergenceBenchmark(dataset="wikitext", config="wikitext-2-raw-v1", split="test", max_tokens=1000),
]
QUANTIZATION_CONFIGS: dict[str, list[str]] = {
    # "default": [],
    "bf16": ["++model.dtype_override=bf16"],
    "8da4w_default": ["++quantization.qmode=8da4w"],
    **{
        f"8da4w_gs{lin_gs}_emb8_{emb_gs}": [
            "++quantization.qmode=8da4w",
            f"++quantization.group_size={lin_gs}",
            f"++quantization.embedding_quantize='8,{emb_gs}'",
        ]
        for emb_gs in (0, 32, 64)
        for lin_gs in (0, 32, 64)
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # fmt: off
    parser.add_argument("--executorch-dir", type=Path, required=True, help="Path to executorch source directory")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "exports", help="Directory to save .pte files and results")

    parser.add_argument("--config-yaml", type=Path, required=True, help="Path to base Hydra config YAML for export (absolute or relative to current working directory)")
    parser.add_argument("--model-class", required=True, help="Model architecture identifier (e.g. 'lfm2_5_1_2b'). Must match a ModelType value in extension/llm/export/config/llm_config.py")
    parser.add_argument("--model-params", type=Path, required=True, help="Path to model params JSON file (absolute or relative to --executorch-dir)")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace repo for tokenizer")
    parser.add_argument("--max-seq-length", type=int, required=True, help="Max sequence length for export")
    parser.add_argument("--max-context-length", type=int, required=True, help="Max context length for export")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (.pth). Required for models that don't auto-download from HuggingFace (e.g. Llama 3.2)")
    parser.add_argument("--metadata", type=str, default=None, help='JSON metadata override, e.g. \'{"get_bos_id":1,"get_eos_ids":[2]}\'. Takes precedence over the config YAML.')

    parser.add_argument("--debug", action="store_true", help="Enable verbose logging during export")
    parser.add_argument("--recompute-exports", action="store_true", help="Re-export even if .pte already exists")
    parser.add_argument("--recompute-benchmarks", action="store_true", help="Re-benchmark even if results already in results.csv")
    parser.add_argument("--discard-exports", action="store_true", help="Delete .pte files after benchmarking")
    # fmt: on

    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    exports_dir: Path = output_dir / f"{args.model_class}"
    results_file: Path = exports_dir / "results.csv"
    plot_file: Path = exports_dir / "results.html"

    config = ExportConfig(
        executorch_dir=args.executorch_dir.resolve(),
        output_dir=exports_dir,
        config_yaml=Path.cwd() / args.config_yaml if not args.config_yaml.is_absolute() else args.config_yaml,
        model_class=args.model_class,
        model_params=args.model_params.resolve(),
        max_seq_length=args.max_seq_length,
        max_context_length=args.max_context_length,
        debug=args.debug,
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        metadata=args.metadata,
    )

    if args.recompute_benchmarks or not results_file.exists():
        df = pd.DataFrame(columns=["name"])
    else:
        df = pd.read_csv(results_file)

    reference_pte: Optional[Path] = None
    reference_model: Optional[ETModelForCausalLMWrapper] = None

    for config_name, overrides in QUANTIZATION_CONFIGS.items():
        print(f"\n{'='*60}\n{config_name}\n{'='*60}")

        try:
            pte = exports_dir / config_name / f"{config_name}.pte"
            needs_benchmark = args.recompute_benchmarks or config_name not in df["name"].values

            if args.recompute_exports or (not pte.exists() and needs_benchmark):
                pte = run_export(config_name, overrides, config)

            if reference_model is None:
                reference_pte = pte
                reference_model = ETModelForCausalLMWrapper.from_local(pte, args.tokenizer)

            if pte.exists():
                print("Sample generation from model:")
                model = ETModelForCausalLMWrapper.from_local(pte, args.tokenizer)
                print((model.text_generate(prompt="In the bleak midwinter", max_seq_len=64)))

            if needs_benchmark:
                metrics = {}
                for benchmark in BENCHMARKS:
                    lm = ETModelForCausalLMWrapper.from_local(pte, args.tokenizer)
                    if isinstance(benchmark, KLDivergenceBenchmark):
                        benchmark.set_reference_model(reference_model.copy())
                    metrics |= benchmark.run(lm)

                df = pd.concat([df, pd.DataFrame([{"name": config_name, **metrics}])], ignore_index=True)
                print(df.to_string(index=False))

            if args.discard_exports and pte != reference_pte:
                shutil.rmtree(pte.parent)

            df.to_csv(results_file, index=False)

        except Exception as e:
            print(f"  [error] {e}")

    if args.discard_exports and reference_pte is not None:
        shutil.rmtree(reference_pte.parent)

    df.to_csv(results_file, index=False)

    if not df.empty:
        metrics_hib: dict[str, bool] = {}
        for benchmark in BENCHMARKS:
            metrics_hib |= benchmark.get_metrics_hib()

        create_plot(
            df,
            metric_registry=metrics_hib,
            save_path=plot_file,
            title=f"{args.model_class} Benchmark Results",
        )


if __name__ == "__main__":
    main()
