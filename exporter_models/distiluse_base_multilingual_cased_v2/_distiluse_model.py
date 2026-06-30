import torch
import torch.nn.functional as F

_SOURCE = "sentence-transformers/distiluse-base-multilingual-cased-v2"
# BERT-convention special tokens (the published .pte prepends them in-graph).
_CLS = 101
_SEP = 102


class DistiluseEmbedder(torch.nn.Module):
    """DistilBERT -> mean pooling -> Dense(768->512, Tanh) -> L2 normalize.

    Two-input contract matching the published .pte: (input_ids, attention_mask).
    CLS/SEP are prepended/appended in-graph, so the caller passes the raw token
    ids. attention_mask is NOT passed to BERT (RNE never pads, so it is all-ones)
    — this avoids the mask-to-4D ops XNNPACK cannot delegate; the real mask is
    still used for mean pooling.
    """

    def __init__(self, bert, dense):
        super().__init__()
        self.bert = bert
        self.dense = dense

    def forward(self, input_ids, attention_mask):
        cls = torch.tensor([[_CLS]], dtype=input_ids.dtype)
        sep = torch.tensor([[_SEP]], dtype=input_ids.dtype)
        input_ids = torch.cat([cls, input_ids, sep], dim=1)
        one = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([one, attention_mask, one], dim=1)

        tokens = self.bert(input_ids=input_ids, attention_mask=None)[0]
        m = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        pooled = (tokens * m).sum(1) / m.sum(1).clamp(min=1e-9)
        projected = self.dense({"sentence_embedding": pooled})["sentence_embedding"]
        return F.normalize(projected, p=2, dim=1)


def load_embedder() -> DistiluseEmbedder:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(_SOURCE, device="cpu")
    bert = st[0].auto_model.cpu()
    dense = st[2].cpu()
    return DistiluseEmbedder(bert, dense).eval()
