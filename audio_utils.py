import os
import hashlib
import time
import pandas as pd
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

def convert_flac_to_wav(input_path: str, output_dir: str) -> str:
    start_time = time.time()
    with open(input_path, 'rb') as f:
        raw_bytes = f.read()
    hash_id = hashlib.sha256(raw_bytes).hexdigest()
    audio = AudioSegment.from_file(input_path, format="flac")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{hash_id}.wav")
    audio.export(output_path, format="wav")
    latency = time.time() - start_time
    print(f"{hash_id}.wav | {latency:.2f}s")
    return f"{hash_id}.wav"  

if __name__ == "__main__":
    input_dir = "train_subset"
    output_dir = "processed_train_subset"
    tsv_path = os.path.join(input_dir, "train_subset.tsv")
    new_tsv_path = "updated_file.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    updated_rows = []

    for _, row in df.iterrows():
        original_file = row["path"]
        transcript = row["sentence"]
        input_path = os.path.join(input_dir, original_file)

        if not os.path.exists(input_path):
            print(f"Skipping missing file: {input_path}")
            continue

        try:
            new_filename = convert_flac_to_wav(input_path, output_dir)
            updated_rows.append({"path": new_filename, "sentence": transcript})
        except Exception as e:
            print(f"Error converting {original_file}: {e}")

    pd.DataFrame(updated_rows).to_csv(new_tsv_path, sep="\t", index=False)