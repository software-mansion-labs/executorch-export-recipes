import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_kokoro_model():
    pkg_init = _MODEL_DIR / "_kokoro_model" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "kokoro_kokoro_model_xnnpack", pkg_init,
        submodule_search_locations=[str(_MODEL_DIR / "_kokoro_model")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kokoro_kokoro_model_xnnpack"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> None:
    # The Polish and German .pte published for this backend were produced
    # outside this repo, so only the base weights are exported here. The Core ML
    # exporter covers all three sizes.
    size = config.get("size") or "standard"
    if size != "standard":
        raise NotImplementedError(
            f"the 'xnnpack' kokoro export covers the standard weights only, got size {size!r}"
        )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    _load_kokoro_model()
    from kokoro_kokoro_model_xnnpack.model import KModel

    class DurationPredictorWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.bert = model.bert
            self.bert_encoder = model.bert_encoder
            self.text_encoder_module = model.predictor.text_encoder
            self.lstm = model.predictor.lstm
            self.duration_proj = model.predictor.duration_proj
            self.lstm_padding = model.lstm_padding

        def forward(self, input_ids: torch.LongTensor, text_mask: torch.BoolTensor,
                    s: torch.FloatTensor, speed: torch.Tensor):
            if self.lstm_padding:
                pad_len = self.lstm_padding - input_ids.shape[-1]
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
                text_mask = torch.nn.functional.pad(text_mask, (0, pad_len), value=False)

            bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            d = self.text_encoder_module(d_en, s, torch.tensor(input_ids.shape[-1]), ~text_mask)

            x, _ = self.lstm(d)

            duration = self.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

            return pred_dur, d

    class Synthesizer(torch.nn.Module):
        def __init__(self, model, duration_padding=None):
            super().__init__()
            self.shared = model.predictor.shared
            self.F0_blocks = model.predictor.F0
            self.F0_proj = model.predictor.F0_proj
            self.N_blocks = model.predictor.N
            self.N_proj = model.predictor.N_proj
            self.lstm_dur_padding = duration_padding
            self.text_encoder = model.text_encoder
            self.decoder = model.decoder

        def forward(self, input_ids: torch.LongTensor,
                    text_mask: torch.BoolTensor,
                    indices: torch.LongTensor,
                    d: torch.FloatTensor,
                    ref_s: torch.FloatTensor):
            ref = ref_s[:, :128]
            s = ref_s[:, 128:]

            input_size = input_ids.shape[-1]

            pred_aln_trg = torch.zeros((input_size, indices.shape[0]), dtype=torch.float32)
            pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1.0
            pred_aln_trg = pred_aln_trg.unsqueeze(0)
            en = d.transpose(-1, -2) @ pred_aln_trg

            x = en.transpose(-1, -2)
            torch._check_is_size(x.shape[1])

            if self.lstm_dur_padding is not None:
                original_seq_len = x.shape[1]
                torch._check_is_size(original_seq_len)
                pad_amount = self.lstm_dur_padding - original_seq_len
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))

            x, _ = self.shared(x)

            if self.lstm_dur_padding is not None:
                x = x[:, :original_seq_len, :]

            F0 = x.transpose(-1, -2)
            for block in self.F0_blocks:
                F0 = block(F0, s)
            F0 = self.F0_proj(F0).squeeze(1)

            N = x.transpose(-1, -2)
            for block in self.N_blocks:
                N = block(N, s)
            N = self.N_proj(N).squeeze(1)

            t_en = self.text_encoder(input_ids, ~text_mask)
            asr = t_en @ pred_aln_trg

            return self.decoder(asr, F0, N, ref)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dp_inputs_dict = {
        "inp-32": (
            torch.randint(0, 178, size=(1, 32)),
            torch.ones((1, 32), dtype=torch.bool),
            torch.randn(size=(1, 128)),
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "inp-64": (
            torch.randint(0, 178, size=(1, 64)),
            torch.ones((1, 64), dtype=torch.bool),
            torch.randn(size=(1, 128)),
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "inp-128": (
            torch.randint(0, 178, size=(1, 128)),
            torch.ones((1, 128), dtype=torch.bool),
            torch.randn(size=(1, 128)),
            torch.tensor([1.0], dtype=torch.float32),
        ),
    }
    input_name_mappings = {"inp-32": 32, "inp-64": 64, "inp-128": 128}

    predictors = {}
    for name, n_tokens in input_name_mappings.items():
        model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=None)
        wrapper = DurationPredictorWrapper(model)
        wrapper.eval()
        inputs = dp_inputs_dict[name]
        input_ids = inputs[0]
        input_ids[0][0] = 0
        input_ids[0][-1] = 0
        ep = torch.export.export(wrapper, inputs)
        predictors[n_tokens] = ep

    dp_program = to_edge_transform_and_lower(
        {f"forward_{n}": ep for n, ep in predictors.items()},
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    dp_path = out / "duration_predictor_std.pte"
    dp_path.write_bytes(dp_program.buffer)

    syn_inputs = (
        torch.randint(0, 178, size=(1, 128)),
        torch.ones((1, 128), dtype=torch.bool),
        torch.randint(0, 2, (296,), dtype=torch.long),
        torch.randn(size=(1, 128, 640)),
        torch.randn(size=(1, 256)),
    )

    def remove_weight_norms(synthesizer):
        remove = torch.nn.utils.parametrize.remove_parametrizations
        modules = list(synthesizer.F0_blocks) + list(synthesizer.N_blocks)
        for module in modules:
            convs = [module.conv1, module.conv2]
            if hasattr(module.pool, "weight"):
                convs.append(module.pool)
            if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
                convs.append(module.conv1x1)
            for conv in convs:
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)

        for module in synthesizer.text_encoder.cnn:
            conv = module[0]
            _ = conv.weight
            remove(conv, "weight", leave_parametrized=True)

        for module in synthesizer.decoder.generator.resblocks:
            for conv in list(module.convs1) + list(module.convs2):
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for module in synthesizer.decoder.generator.noise_res:
            for conv in list(module.convs1) + list(module.convs2):
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for conv_t in synthesizer.decoder.generator.ups:
            _ = conv_t.weight
            remove(conv_t, "weight", leave_parametrized=True)
        _ = synthesizer.decoder.generator.conv_post.weight
        remove(synthesizer.decoder.generator.conv_post, "weight", leave_parametrized=True)

        modules = list(synthesizer.decoder.decode) + [synthesizer.decoder.encode]
        for module in modules:
            convs = [module.conv1, module.conv2]
            if hasattr(module.pool, "weight"):
                convs.append(module.pool)
            if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
                convs.append(module.conv1x1)
            for conv in convs:
                _ = conv.weight
                remove(conv, "weight", leave_parametrized=True)
        for conv in [synthesizer.decoder.F0_conv, synthesizer.decoder.N_conv, synthesizer.decoder.asr_res[0]]:
            _ = conv.weight
            remove(conv, "weight", leave_parametrized=True)

    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=None)
    synthesizer = Synthesizer(model, duration_padding=None)
    synthesizer.eval()
    remove_weight_norms(synthesizer)

    syn_ep = torch.export.export(synthesizer, syn_inputs)
    syn_program = to_edge_transform_and_lower(
        syn_ep,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    syn_path = out / "synthesizer_std.pte"
    syn_path.write_bytes(syn_program.buffer)
