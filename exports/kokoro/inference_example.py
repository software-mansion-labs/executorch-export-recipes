from executorch.runtime import Runtime
from huggingface_hub import hf_hub_download
from model.postprocessing import find_voice_bound
from model.temporal_scaling import scale
from misaki import en, espeak
import numpy as np
import os
import torch

runtime = Runtime.get()

# Inference example
#
# This demo script focuses on showcasing how to utilize exported
# models (.pte) and create a full Kokoro pipeline to transform
# the input text into output audio vector.

if __name__ == "__main__":
  # Constants
  REPO_ID = "software-mansion/react-native-executorch-kokoro"
  CACHE_DIR = "."
  MODEL_VOCAB = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19,
    "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23, "A": 24, "I": 25, "O": 31, "Q": 33,
    "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45,
    "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54,
    "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62,
    "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70,
    "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81,
    "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99,
    "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113,
    "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123,
    "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132,
    "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142,
    "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162,
    "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177
  }

  # Define paths to Kokoro submodules:
  # - DurationPredictor
  # - Synthesizer
  # If selected paths do not exist, the script will attempt to download 
  # the files from the huggingface repository.
  MODEL_FILEPATHS = {
    "duration_predictor": ...,
    "synthesizer": ...
  }

  # Download missing model files if not present in selected paths
  for name, path in MODEL_FILEPATHS.items():
      filename = name + ".pte"
      if path is None or path is ... or not os.path.exists(path):
          user_input = input(f"{filename} not found. Download from HuggingFace? [y/N]: ").strip().lower()
          if user_input == "y":
              new_path = hf_hub_download(REPO_ID, "xnnpack/medium/" + filename, cache_dir=CACHE_DIR)
              MODEL_FILEPATHS[name] = new_path
          else:
              print(f"Skipping download for {filename}.")
              MODEL_FILEPATHS[name] = None

  # Load .pte models
  # As it stands for now, the synthesizer model is exported with dynamic shapes, 
  # allowing to process up to 128 tokens in a single run.
  # The durationPredictor model however, is exported with semi-dynamic shapes:
  # but you can use method 'forward_128', which accepts any number of tokens
  # from 1 up to 128.
  MAX_INPUT_SIZE = 128
  MAX_DURATION = 296
  METHOD_NAME = f"forward_{MAX_INPUT_SIZE}"

  duration_predictor = runtime.load_program(MODEL_FILEPATHS["duration_predictor"]).load_method(METHOD_NAME)
  synthesizer = runtime.load_program(MODEL_FILEPATHS["synthesizer"]).load_method("forward")

  # Load sample voice from HF
  VOICE_PATH = hf_hub_download(REPO_ID, "voices/af_heart.bin", cache_dir=CACHE_DIR)
  with open(VOICE_PATH, "rb") as f:
    voice_bytes = f.read()
    voice_array = np.frombuffer(voice_bytes, dtype=np.float32).reshape(510, 1, 256)
    voice = torch.from_numpy(voice_array)
  print(voice.shape)

  # Define and phonemize the input text
  # Kokoro does not take plain text as an input - it takes phonemes (spoken representation).
  # This means you must use some phonemizer such as Misaki in this case.
  INPUT_TEXT = "Kokoro is such a wonderful model!"
  USE_FALLBACK = True

  fallback = espeak.EspeakFallback(british=False) if USE_FALLBACK else None
  phonemizer = en.G2P(trf=False, fallback=fallback, unk='')

  INPUT_PHONEMES = phonemizer(INPUT_TEXT)[0]
  print("Input phonemes:", INPUT_PHONEMES)

  # Kokoro inference starts
  # -----------------------
  # Token processing
  input_ids = [MODEL_VOCAB.get(p) for p in INPUT_PHONEMES if MODEL_VOCAB.get(p) is not None]
  input_length = min(len(input_ids) + 2, MAX_INPUT_SIZE)
  # Pad or truncate tokens
  input_ids = ([0] + input_ids[:input_length-2] + [0])
  input_tokens = torch.tensor([input_ids], dtype=torch.int64)
  # Model inputs preparation
  voice_vec = voice[len(INPUT_PHONEMES) - 1]
  v_ref = voice_vec[:, :128]
  v_style = voice_vec[:, 128:]
  text_mask = torch.ones((1, input_length), dtype=torch.bool)
  speed = torch.tensor([1.0], dtype=torch.float32)

  # 1. Duration Predictor
  pred_dur, d = duration_predictor.execute((input_tokens, text_mask, v_style, speed))

  # Cut result back to the original shape
  pred_dur = pred_dur[:input_length]
  d = d[:, :input_length, :]

  # Native duration scaling
  total_dur = pred_dur.sum().item()
  if total_dur > MAX_DURATION:
    pred_dur = scale(pred_dur, MAX_DURATION)
    total_dur = MAX_DURATION
  indices = torch.repeat_interleave(torch.arange(input_length), pred_dur.view(-1))[:total_dur]

  # Calculate efficient duration to trim silence
  first_index = None
  for i, val in enumerate(indices):
      if val.item() == input_length:
          first_index = i
          break
  eff_dur = first_index if first_index is not None else MAX_DURATION

  # 2. Synthesizer
  audio = synthesizer.execute((input_tokens, text_mask, indices, d, voice_vec))[0]

  # Post-processing
  # Crop the audio to remove silent fragments from both ends.
  v_start = find_voice_bound(audio, from_end=False)
  v_end = find_voice_bound(audio, from_end=True)
  final_audio = audio[v_start:v_end]

  print("Inference succeeded!")
  print("Result:", final_audio)
