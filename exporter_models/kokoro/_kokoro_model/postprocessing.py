import torch

# Constants
MOVING_AVERAGE_K = 20
THRESHOLD = 0.01

# Value evaluation
def evaluate(val: float):
    val = abs(val)
    return val if val >= THRESHOLD else 0.0

# Find voice end or start depending on direction
def find_voice_bound(audio: torch.Tensor, from_end: bool = True):
    length = audio.shape[0]

    if from_end:
        moving_sum = 0.0
        moving_count = 0
        for i in range(length - 1, MOVING_AVERAGE_K - 2, -1):
            moving_sum += evaluate(audio[i])
            if i + MOVING_AVERAGE_K < length:
                moving_sum -= evaluate(audio[i + MOVING_AVERAGE_K])
            moving_count = min(MOVING_AVERAGE_K, moving_count + 1)
            avg = moving_sum / moving_count
            if avg >= THRESHOLD:
                return i
    else:
        moving_sum = 0.0
        moving_count = 0
        for i in range(MOVING_AVERAGE_K - 1, length):
            moving_sum += evaluate(audio[i])
            if i - MOVING_AVERAGE_K >= 0:
                moving_sum -= evaluate(audio[i - MOVING_AVERAGE_K])
            moving_count = min(MOVING_AVERAGE_K, moving_count + 1)
            avg = moving_sum / moving_count
            if avg >= THRESHOLD:
                return i
    return None

