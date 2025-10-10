import torch 
from  torch import Tensor
from typing import List,Dict 

'''
  Postprocess is very similair, to what is done in silero-vad 
  `get_speech_timestamps()` method. 
  Postprocess as an input takes the tensor of probabilites 
  of voice NOT detected, and returns the list of segments 
  (dictionary with keys "start", "end" reffering to sample number)
'''
class VADPostprocess:
  def __init__(self):
    self.SAMPLE_RATE_MS = 16  # 16k samples per second so 16 per ms
    self.WINDOW_SIZE = 10 # each score can be understood 
                          # as the probability of silence in the 10ms time period
    self.THRESHOLD = 0.5
    self.MIN_SPEECH_DURATION: int = 25   #250 MS
    self.MIN_SILENCE_DURATION: int = 10 #100 MS
    self.SPEECH_PAD: int = 3 #30 ms
    self.NEG_THRESHOLD: float = None

  def forward(self, scores: torch.Tensor):
    scores = 1 - scores #change scores from prob of silence to prob of speech
    triggered = False
    speech_segments = []
    start_segment = -1 
    end_segment = -1 
    potential_start = -1
    potential_end = -1 
    for i,score_tensor in enumerate(scores):
      score = score_tensor.item()
      if not triggered:
        if score >= self.THRESHOLD:
          if potential_start == -1: 
            potential_start = i 
          elif i - potential_start >= self.MIN_SPEECH_DURATION: 
            triggered = True 
            start_segment = potential_start 
        else: #score < self.THRESHOLD 
          potential_start = -1 
      else: #triggered 
        if score < self.THRESHOLD:
          if potential_end == -1:
            potential_end = i 
          elif i - potential_end >= self.MIN_SILENCE_DURATION:
            triggered = False 
            end_segment = potential_end
            speech_segments.append({"start" : start_segment, "end" : end_segment})
        else: #score >= self.THRESHOLD 
          potential_end = -1 

    #if end on voice add last speech segment
    if triggered:
      end_segment = len(scores) -1 
      speech_segments.append({"start" : start_segment, "end" : end_segment})
    for segment in speech_segments:
      segment["start"] = max(segment["start"] - self.SPEECH_PAD, 0)
      segment["end"] = min(segment["end"] + self.SPEECH_PAD, len(scores) -1)

    #convert to samples
    for segment in speech_segments:
      segment["start"] = segment["start"] * self.SAMPLE_RATE_MS*self.WINDOW_SIZE
      segment["end"] = segment["end"]  * self.SAMPLE_RATE_MS*self.WINDOW_SIZE
    return speech_segments 


