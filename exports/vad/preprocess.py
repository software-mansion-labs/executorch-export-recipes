import torch 
from  torch import Tensor
from typing import Tuple

MILLISECONDS_TO_SECONDS = 0.001

def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Returns:
        Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0: #true
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            # pad is negative so we want to trim the waveform at the front
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    sizes = (m, window_size)
    return waveform.as_strided(sizes, strides)

def _get_waveform_and_window_properties(
    waveform: Tensor,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    round_to_power_of_two: bool,
) -> Tuple[Tensor, int, int, int]:
    r"""Gets the waveform and window properties"""
    channel = 0
    waveform = waveform[channel, :]  
    sample_frequency = 16_000.0 
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    # next power of 2 = 512 
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), "choose a window size {} that is [2, {}]".format(
        window_size, len(waveform)
    )
    assert padded_window_size % 2 == 0, (
        "the padded `window_size` must be divisible by two." " use `round_to_power_of_two` or change `frame_length`"
    )
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: Tensor,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    snip_edges: bool,
    preemphasis_coefficient: float,
) -> Tuple[Tensor, Tensor]:
    r"""Gets a window and its log energy

    Returns:
        (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
    """
    device, dtype = waveform.device, waveform.dtype

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
    strided_input = strided_input - row_means

    if preemphasis_coefficient != 0.0: #always true = 0.97 
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
            0
        )  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function =  torch.hamming_window(window_size, periodic=False, alpha=0.54, 
                                            beta=0.46, device=device,
                                            dtype=dtype).unsqueeze(0)  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
        ).squeeze(0)

    return strided_input

def get_windowing(
      waveform: Tensor,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    preemphasis_coefficient: float = 0.97,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    
) -> Tensor: 

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, sample_frequency, frame_shift, frame_length, round_to_power_of_two
    )

    strided_input= _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        snip_edges,
        preemphasis_coefficient,
    )
    return strided_input

'''
   Preprocessing outside of the model is standard audio framing.
  We take the waveform with shape (1, len) and segment it into overlapping
  windows (frames) to extract features over time.

  - Window size: 25ms. The duration of each frame.
  - Window shift: 10ms. The amount of time between the start of consecutive frames.

  CONFIGURATION:
  
  SNIP_EDGES = False: 
    If False, the waveform is padded with zeros at the ends. This allows frames
    to be centered on the first and last samples of the signal, ensuring that:

      number_frames = (signal_length + hop_length - 1) // hop_length 
    
    If True, any partial frames at the edges that
    would require padding are discarded.

  ROUND_TO_POWER_OF_TWO = True:
    If True, each frame is zero-padded to the next highest power-of-two length
    (in our case, a 400-sample frame (25ms from 16kHz) audio becomes 512 samples). This is a computational optimization, as the FFT algorithm is most efficient with input sizes of 2^n.
'''
class VADPreprocess():
  def __init__(self):
    self.FRAME_LEN = 25.0
    self.FRAME_SHIFT = 10.0    
    self.SAMPLE_RATE = 16000.0
    self.PREEMPHASIS_COEFF = 0.97
    self.ROUND_TO_POWER_OF_TWO = True
    self.SNIP_EDGES = False 

  def forward(self, waveform : torch.tensor):
    strided_input = get_windowing(waveform,self.FRAME_LEN, 
                                  self.FRAME_SHIFT, self.PREEMPHASIS_COEFF,
                                  self.ROUND_TO_POWER_OF_TWO, self.SAMPLE_RATE,
                                  self.SNIP_EDGES)
    return strided_input 
  