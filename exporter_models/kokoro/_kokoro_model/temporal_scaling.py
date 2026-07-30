import torch

# A simple heuristic to "scale" integer-based indice duration tensor to match the target duration
def scale(pred_dur: torch.Tensor, target_len: int,
          up: bool = True) -> torch.Tensor:
    # NOTE: a is a 1D tensor of shape [N]
    factor = target_len / pred_dur.sum().item()

    x = pred_dur * factor
    b = torch.floor(x).long() if up else torch.ceil(x).long()

    remaining = abs(target_len - b.sum().item())

    diff: torch.Tensor = torch.abs(x - b)
    diff_table = [(idx, value) for idx, value in enumerate(diff)]
    diff_table.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining):
      b[diff_table[i][0]] += 1 if up else -1
    
    return b
