import os
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset, Audio
import hashlib
import pandas as pd

# Constants
HF_DATASET = "mozilla-foundation/common_voice_11_0"
LANG = "de"
SPLIT = "train[:1%]"  # or "validation", etc.
OUTPUT_DIR = "processed_train_subset"
TSV_SAVE_PATH = "updated_file.tsv"

# Resampler to 16kHz mono
resampler = T.Resample(orig_freq=48000, new_freq=16000)  # adjust if needed

# Load dataset
dataset = load_dataset(HF_DATASET, LANG, split=SPLIT)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Processing loop
os.makedirs(OUTPUT_DIR, exist_ok=True)
new_rows = []

for example in dataset:
    audio_array = example["audio"]["array"]
    transcript = example["sentence"]
    orig_sr = example["audio"]["sampling_rate"]

    # Ensure mono 16kHz
    if orig_sr != 16000:
        audio_tensor = torchaudio.functional.resample(
            torch.tensor(audio_array), orig_freq=orig_sr, new_freq=16000
        )
    else:
        audio_tensor = torch.tensor(audio_array)

    # Save as WAV with hash-based name
    hash_id = hashlib.sha256(audio_tensor.numpy().tobytes()).hexdigest()
    wav_path = os.path.join(OUTPUT_DIR, f"{hash_id}.wav")
    torchaudio.save(wav_path, audio_tensor.unsqueeze(0), 16000)

    new_rows.append({"path": f"{hash_id}.wav", "sentence": transcript})
    print(f"Saved: {wav_path}")

# Save TSV
pd.DataFrame(new_rows).to_csv(TSV_SAVE_PATH, sep="\t", index=False)
print(f"\nSaved updated TSV to {TSV_SAVE_PATH}")
