from postprocess import VADPostprocess
from preprocess import VADPreprocess
from vad_model import VAD
from executorch.runtime import Verification, Runtime, Program, Method
from pathlib import Path
import torchaudio
import torch

def load_executorch_model():
  et_runtime: Runtime = Runtime.get()
  program: Program = et_runtime.load_program(
      Path("fsmn-vad-dynamic-6.pte"),
      verification=Verification.Minimal,
  )
  print("Program methods:", program.method_names)
  forward_executorch: Method = program.load_method("forward")
  return forward_executorch

def waveform_from_audio(audio_path: str) -> torch.tensor:
    waveform, _ = torchaudio.load(audio_path)
    return waveform

class VADPipeline():
  def __init__(self, 
               preprocess = VADPreprocess(),
               postprocess = VADPostprocess(),
               forward : Method = load_executorch_model()):
    self.MIN_SIZE = 16_000
    self.MAX_SIZE = 160_000
    self.preprocess = preprocess
    self.postprocess = postprocess 
    self.forward = forward 

  def join_segments(self,segment_list):
    shift = 0 
    final_list = []
    for segments in segment_list:
      for segment in segments:
        segment["start"] = segment["start"] + shift
        segment["end"] = segment["end"] + shift
        final_list.append(segment)
      shift+=self.MAX_SIZE
    return final_list

  def run(self,audio_path : str):
    '''
      raw model accepts input betwen 1-10 seconds (which is 16 000 - 160 000 samples).
      If the audio is longer we cut it into pieces. If the remainder is < 16 00 samples
      we pad it with zeros. 
    '''
    result = []
    waveform = waveform_from_audio(audio_path)
    original_size = waveform.shape[1]
    remainder = original_size % self.MAX_SIZE
    if remainder < self.MIN_SIZE:
      waveform = torch.nn.functional.pad(waveform, (0, self.MIN_SIZE - remainder,0,0))
    waveform_chunks = torch.split(waveform, self.MAX_SIZE, dim=1)

    for wave in waveform_chunks[:-1]:
      strided_input = self.preprocess.forward(wave)
      scores = self.forward.execute((strided_input,))[0][0,:,0]
      segments = self.postprocess.forward(scores)
      result.append(segments)   
    
    # process the last chunk seperatly as it could be padded 
    wave = waveform_chunks[-1]
    strided_input = self.preprocess.forward(wave)
    scores = self.forward.execute((strided_input,))[0][0,:remainder,0]
    segments = self.postprocess.forward(scores)
    result.append(segments)
    return self.join_segments(result)

    
  


  

