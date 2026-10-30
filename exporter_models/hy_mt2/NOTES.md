# hy_mt2

Tencent Hy-MT2-1.8B (Apache-2.0, 33 languages + dialects, 38 codes), exported
for translation through export_llama. HunYuan dense v1 = Llama + per-head q/k
RMSNorm after RoPE (`use_qk_norm`, `qk_norm_before_rope=False`). RoPE is
"dynamic" with alpha 1000 and factor 1.0, i.e. a fixed base
1e4 * 1000^(128/126) baked into `rope_theta`. Eager parity with HF
transformers (fp32, 38-token prompt): max |logit diff| 0.0.

Prompt it as the model card does, no system prompt. English prompt, or the
Chinese prompt with Chinese language names when either side is Chinese:

```
<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into {Tgt}. Note that you should only output the translated result without any additional explanation:\n\n{text}<｜hy_Assistant｜>
<｜hy_begin▁of▁sentence｜><｜hy_User｜>将以下文本翻译为{目标语言}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}<｜hy_Assistant｜>
```

## Quality and speed (host, 2026-10-29)

Greedy, first 50 sentences, sacrebleu (zh tokenizer for ->zh). Runtime: fork
f7bd426 pybindings (what v0.10.7-libs ship); AOT: ET 1.4.1 + fork 042c46e
python. Host ms are indicative only.

| variant | size | en->de (WMT13) | en->zh (WMT22) | zh->en (WMT22) | decode ms/token |
| --- | --- | --- | --- | --- | --- |
| fp32 eager (HF) | - | 24.80 | 43.28 | 26.47 | - |
| xnnpack 8da4w gs32 (default) | 1.17 GB | 25.37 | 42.18 | 25.62 | 14.4 |
| xnnpack 8da8w | 2.05 GB | 26.70 | 42.37 | 27.08 | 21.5 |
| mlx int8 | 1.85 GB | 25.65 | 42.83 | 25.76 | 17.0 |
| EuroLLM-1.7B xnnpack 8da8w (published) | 1.66 GB | 27.83 | 40.74 | 25.10 | 16.7 |

chrF: 8da4w 55.70 / 36.18 / 56.85, 8da8w 57.43 / 36.45 / 57.21, mlx 56.90 /
36.76 / 57.41, EuroLLM 57.60 / 35.07 / 52.96.

Unlike EuroLLM (6-9 BLEU lost at 4-bit), 8da4w costs 0.2-1.5 BLEU and stays
within ~1 BLEU of fp32, for 43% less size and 1.5x faster decode, so it is the
XNNPACK default; 8da8w is published for quality. No empty or runaway output
in any Hy-MT2 run (EuroLLM ran away on 1 of the en->de sentences).

## On device (2026-10-30)

react-native-executorch PR #1572 (52e96e87d7, native libs v0.11.0), `apps/nlp` Release,
`createLLMTranslator` with the published HF v0.11.0 files, first 10 sentences of each
eval set above, one run each. "Identical" = same text as the host greedy output of the
same .pte; ms are wall time per sentence (prefill + decode) as the app sees it.

| device | variant | load | identical en-de / en-zh / zh-en | ms/sentence en-de / en-zh / zh-en |
| --- | --- | --- | --- | --- |
| Galaxy S26 Ultra | xnnpack 8da4w | 1.4 s | 7 / 8 / 7 | 1041 / 433 / 712 |
| Galaxy S26 Ultra | xnnpack 8da8w | 3.5 s | 5 / 4 / 5 | 1603 / 729 / 1230 |
| iPhone 16 | mlx int8 | 0.3 s | 10 / 10 / 10 | 1940 / 776 / 1370 |
| iPhone 16 | xnnpack 8da4w | 2.6 s | 3 / 4 / 5 | 1277 / 725 / 1152 |

Non-identical outputs are near-tie flips into equally valid wording (char similarity to
host 0.92-0.99; e.g. comma placement, synonyms); XNNPACK's int8 kernels differ per CPU,
MLX on the iPhone reproduces the M-series host exactly. No errors, empty or runaway
outputs. The published tokenizer.json loads and tokenizes correctly on the v0.11.0 libs.
On the iPhone 8da4w takes 7-34% less time per sentence than MLX int8 and is 37% smaller.

## Tokenizer (`_tokenizer.py`)

The HFTokenizer in RNE's native libs (pytorch/tokenizers fork ef00dcc4)
cannot read upstream's tokenizer.json: it rejects the empty `Sequence`
normalizer, and treats a `Split` (Regex, `Isolated`, `invert: false`) as "keep
only the matches", so the digit and CJK splits drop all other text and the
last split drops bare digit runs. Both fixed upstream in
meta-pytorch/tokenizers#225 (0d88e869), after 0.10.7-libs.

`_tokenizer.py` rewrites the file so the old loader gives the same ids:
normalizer `null`, and each split regex gets an alternative matching the text
between its matches. Checked with the fork's C++ HFTokenizer against HF
`tokenizers` on 71,443 strings (17 WMT test sets src+ref, 3000 mixed-script
multi-line joins, edge cases): identical ids and token-by-token streamed decode
on all but `x\r\ny`. The CR loss is the C++ BPE dropping `\r` (same with a
ByteLevel-only pre-tokenizer), unrelated to the rewrite; RNE trims each line.

It also adds `chat_template` (upstream ships `chat_template.jinja`) to
tokenizer_config.json, where RNE reads it.

## Export env

The model's uv project pins the fork like eurollm's, which is not uv-buildable
here (ET's source-dir name + torch-nightly C++ break). Exported with
`PYTHONPATH=src ~/et-aot/venv-fork/bin/python -m exporter.cli export hy_mt2
--size 1_8b --backend <b> --all-variants`, the env EuroLLM was exported with.
MLX .pte files need a fork-built runtime to execute on the host: stock 1.4.1
pybindings fail with `[scatter_add_axis] Received invalid axis`.

## Not exported

Vulkan: EuroLLM's int8 Vulkan LLM path was both slow (no M=1 GEMV) and wrong
(fp16 RMSNorm overflow) on the S26 Ultra; nothing about Hy-MT2 changes that.
