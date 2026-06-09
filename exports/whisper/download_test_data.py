import os
import argparse
from datasets import load_dataset
import soundfile as sf
import json

def download_and_save_data(n_samples=10, output_dir="test_data"):
    """
    Downloads N samples from LibriSpeech validation-clean and saves them as audio files and JSON metadata.
    """
    print(f"Loading LibriSpeech 'clean' validation split for {n_samples} samples...")
    # Use streaming=True to fetch ONLY the requested samples without downloading 
    # the large .parquet files. To avoid "Bad file descriptor" issues, 
    # we iterate carefully with a proper take() limit.
    ds = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Take only the required samples from the stream
    for i, sample in enumerate(ds.take(n_samples)):
        audio_data = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        text = sample["text"]
        speaker_id = sample["speaker_id"]
        chapter_id = sample["chapter_id"]
        id_ = sample["id"]

        # Define filenames (no suffixes, just sample_x.wav and sample_x.json)
        audio_filename = f"sample_{i}.wav"
        audio_path = os.path.join(output_dir, audio_filename)

        # Save audio file
        sf.write(audio_path, audio_data, sampling_rate)

        # Prepare metadata
        metadata = {
            "audio_file": audio_filename,
            "transcription": text,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "id": id_
        }

        # Save metadata to a JSON file for each sample
        meta_filename = f"sample_{i}.json"
        meta_path = os.path.join(output_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved sample {i+1}: {audio_path}")

    print(f"Finished downloading and saving {n_samples} samples.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LibriSpeech samples.")
    parser.add_argument("-n", "--num_samples", type=int, default=10, help="Number of samples to download")
    parser.add_argument("-o", "--output_dir", type=str, default="test_data", help="Directory to save the files")
    
    args = parser.parse_args()
    download_and_save_data(args.num_samples, args.output_dir)
