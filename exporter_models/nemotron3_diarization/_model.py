"""Export boundaries for pinned HF Nemotron 3 Diarization, with external streaming state."""
from pathlib import Path
import torch
from torch import nn

SOURCE_REPO = 'nvidia/Nemotron-3-Diarization'
SOURCE_REVISION = '52a432672732425e4c19765b9fec9b5810e6f4ba'
MAX_AUDIO = 160000
MAX_EMBEDS = 800


def load_source(local_path=None):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    kwargs = {'token': False} if local_path else {'revision': SOURCE_REVISION, 'token': False}
    path = local_path or SOURCE_REPO
    processor = AutoProcessor.from_pretrained(path, **kwargs)
    model = AutoModelForAudioFrameClassification.from_pretrained(path, attn_implementation='eager', **kwargs).eval()
    return model, processor


class Features(nn.Module):
    """The original feature extractor's tensor operations, fixed capacity with true-length masking.

    Adapted from transformers 5.20.0 NemotronAsrStreamingFeatureExtractor.
    Copyright 2026 The HuggingFace Inc. team, Apache-2.0. See THIRD_PARTY_LICENSE.
    No feature normalization, resampling, or numerical shortcut is introduced.
    """
    def __init__(self, extractor):
        super().__init__()
        self.register_buffer('mel_filters', extractor.mel_filters)
        self.register_buffer('window', torch.hann_window(400, periodic=False))

    def forward(self, audio, length):
        x = audio.unsqueeze(0)
        x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)
        x = x.masked_fill(torch.arange(MAX_AUDIO)[None, :] >= length[:, None], 0.)
        stft = torch.stft(x, 512, hop_length=160, win_length=400, window=self.window,
                         return_complex=True, pad_mode='constant', center=True)
        magnitudes = torch.view_as_real(stft).pow(2).sum(-1).sqrt().pow(2)
        features = torch.log(self.mel_filters @ magnitudes + 2**-24).transpose(1, 2)
        mask = torch.arange(1001)[None, :] < (length // 160)[:, None]
        return features * mask.unsqueeze(-1)


class Embed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedder = model.model.audio_tower.embedder

    def forward(self, features):
        return self.embedder(features)


class Encode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.classifier = model.classifier

    def forward(self, embeddings, valid):
        # The original bidirectional attention mask masks padded keys. Supplying
        # its broadcastable 4D form avoids tracing HF's general mask dispatcher.
        is_valid = valid != 0
        mask = torch.where(is_valid[:, None, None, :], 0., torch.finfo(torch.float32).min)
        output = self.model(inputs_embeds=embeddings, attention_mask=mask,
                            position_ids=torch.arange(embeddings.shape[1])[None, :])
        probabilities = self.classifier(output.last_hidden_state).sigmoid()
        pooled = torch.nn.functional.avg_pool1d(probabilities.transpose(1, 2), 8, 8).transpose(1, 2)
        return probabilities, pooled * is_valid.unsqueeze(-1)


class Compress(nn.Module):
    def __init__(self, model):
        super().__init__()
        from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import Nemotron3DiarizationSpeakerCache
        self.policy = Nemotron3DiarizationSpeakerCache(model.config.streaming_config)
        self.register_buffer('silence', model.silence_embeds.detach())

    def forward(self, embeddings, probabilities):
        # Adapted from Transformers 5.20 Nemotron3DiarizationSpeakerCache._compress.
        # Copyright 2026 The HuggingFace Inc. team, Apache-2.0.
        # EXPORT PATCH: functional frame boost preserves the N=265 boundary;
        # the original in-place tail slice creates an N-264 != 1 export guard.
        import math
        policy = self.policy
        batch_size, num_frames, num_speakers = probabilities.shape
        # EXPORT PATCH: finite extreme sentinels serialize through FlatBuffers.
        # All real scores are bounded by the original 0.25 probability clamp;
        # the finite extremes preserve ranking, boosts and rejected-frame detection.
        negative = torch.finfo(torch.float32).min
        positive = torch.finfo(torch.float32).max
        log_probs = torch.log(probabilities.clamp(min=policy.prediction_score_threshold))
        complements = torch.log((1.0 - probabilities).clamp(min=policy.prediction_score_threshold))
        scores = log_probs - complements + complements.sum(dim=-1, keepdim=True) - math.log(0.5)
        speech = probabilities > 0.5
        scores = scores.masked_fill(~speech, negative)
        enough = (scores > 0).sum(dim=1, keepdim=True) >= policy.min_positive_scores
        scores = scores.masked_fill(~(scores > 0) & speech & enough, negative)
        boost = (torch.arange(num_frames) >= policy.speaker_cache_length).to(scores.dtype)
        scores = scores + boost[None, :, None] * policy.latest_frames_score_boost
        scores = policy._boost_scores(scores, policy.num_strong_boosted_frames, -2.0 * math.log(0.5))
        scores = policy._boost_scores(scores, policy.num_weak_boosted_frames, -math.log(0.5))
        scores = torch.nn.functional.pad(scores, (0, 0, 0, policy.num_silence_frames), value=positive)
        silence = self.silence.to(embeddings).view(1, 1, -1).expand(batch_size, 1, -1)
        embeds = torch.cat([embeddings, silence], dim=1)
        probs = torch.nn.functional.pad(probabilities, (0, 0, 0, 1))
        num_scored_frames = num_frames + policy.num_silence_frames
        sentinel = num_scored_frames * num_speakers
        flat_scores = scores.transpose(1, 2).reshape(batch_size, -1)
        topk_scores, topk_indices = torch.topk(flat_scores, policy.speaker_cache_length, dim=1, sorted=False)
        topk_indices = topk_indices.masked_fill(topk_scores == negative, sentinel)
        # EXPORT PATCH: sorting all 264 selected indices with topk is identical
        # to sort.values, which the shipping ET 1.4.1 runtime does not register.
        topk_indices, _ = torch.topk(topk_indices, policy.speaker_cache_length, dim=1, largest=False, sorted=True)
        frame_indices = torch.where(topk_indices == sentinel, num_frames, (topk_indices % num_scored_frames).clamp(max=num_frames))
        batch_indices = torch.arange(batch_size)[:, None]
        return embeds[batch_indices, frame_indices], probs[batch_indices, frame_indices]
