import torch

_SOURCE = "sentence-transformers/distiluse-base-multilingual-cased-v2"


class DistiluseEmbedder(torch.nn.Module):
    """DistilBERT -> mean pooling -> Dense(768->512, Tanh), single input_ids.

    The CoreML/XNNPACK models take only input_ids (the caller passes exactly the
    real tokens, special tokens already added by the tokenizer), so the
    attention mask is implicitly all-ones and derived internally. Output is the
    512-d sentence embedding, matching SentenceTransformer.encode()."""

    def __init__(self, bert, dense):
        super().__init__()
        self.bert = bert
        self.dense = dense

    def forward(self, input_ids):
        mask = torch.ones_like(input_ids)
        tokens = self.bert(input_ids=input_ids, attention_mask=mask)[0]
        m = mask.unsqueeze(-1).float()
        pooled = (tokens * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dense({"sentence_embedding": pooled})["sentence_embedding"]


def load_embedder() -> DistiluseEmbedder:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(_SOURCE, device="cpu")
    bert = st[0].auto_model.cpu()
    dense = st[2].cpu()
    return DistiluseEmbedder(bert, dense).eval()
