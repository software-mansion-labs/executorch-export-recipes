from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch


class KModel(torch.nn.Module):
    """
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    """

    MODEL_NAMES = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
        lstm_padding: int | None = None
    ):
        super().__init__()
        if repo_id is None:
            repo_id = "hexgrad/Kokoro-82M"
            print(
                f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning."
            )
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                config = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config, "r", encoding="utf-8") as r:
                config = json.load(r)
        self.vocab = config["vocab"]
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=config["n_token"], **config["plbert"])
        )
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size, config["hidden_dim"]
        )
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
            lstm_padding=lstm_padding
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
            lstm_padding=lstm_padding
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            disable_complex=disable_complex,
            **config["istftnet"],
        )
        if not model:
            model = hf_hub_download(
                repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id]
            )
        for key, state_dict in torch.load(
            model, map_location="cpu", weights_only=True
        ).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)
        
        self.lstm_padding = lstm_padding

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # ------------------------------------------------------------------------------
        # DurationPredictor (duration_predictor.pte) - returns pred_dur, d, s
        # ------------------------------------------------------------------------------
        input_lengths = torch.tensor(input_ids.shape[-1])

        # # Original code supporting arbitrary batch sizes.
        # text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        # text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)

        # Only support batch size of 1 for export.
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)

        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        # # Force batch size = 1.
        # s = ref_s[128:]
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, ~text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        # ------------------------------------------------------------------------------

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device), pred_dur
        )

        torch._check(indices.shape[0] > 0)
        torch._check_is_size(indices.shape[0])
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        # # Force batch size = 1.
        # pred_aln_trg = pred_aln_trg.to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg

        # ------------------------------------------------------------------------------
        # F0NPredictor (f0n_predictor.pte) - returns F0_pred, N_pred
        # ------------------------------------------------------------------------------
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextEncoderWrapper (text_encoder.pte) - returns t_en, takes only input_ids
        # ------------------------------------------------------------------------------
        t_en = self.text_encoder(input_ids, ~text_mask)
        # ------------------------------------------------------------------------------

        asr = t_en @ pred_aln_trg

        # ------------------------------------------------------------------------------
        # TextDecoderWrapper (text_decoder_16_xxx.pte) - returns audio
        # ------------------------------------------------------------------------------
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        # ------------------------------------------------------------------------------
        
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
        target_tokens: int | None = None
    ) -> Union["KModel.Output", torch.FloatTensor]:
        input_ids = list(
            filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes))
        )
        assert len(input_ids) + 2 <= self.context_length, (
            len(input_ids) + 2,
            self.context_length,
        )

        # Cut the number of tokens (as models are being exported with static input)
        if target_tokens:
            while len(input_ids) < (target_tokens - 2):
                input_ids.append(0)
            input_ids = input_ids[:(target_tokens - 2)]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)

        ref_s = ref_s.to(self.device)

        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio


class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
