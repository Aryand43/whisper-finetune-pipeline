import os
import hashlib
import time
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
    return hash_id

if __name__ == "__main__":
    input_dir = "train_subset"
    output_dir = "processed_train_subset"
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".flac"):
            input_path = os.path.join(input_dir, file_name)
            convert_flac_to_wav(input_path, output_dir)
