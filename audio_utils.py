import os
import json
import wave
import contextlib
from pydub import AudioSegment
import pandas as pd
import webrtcvad
import collections
import logging

logging.basicConfig(level=logging.INFO)

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, wf.getframerate()

def frame_generator(audio, sample_rate, frame_duration_ms):
    n = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_trim(audio_path, vad, sample_rate=16000, frame_duration=30, padding_ms=300):
    pcm_data, _ = read_wave(audio_path)
    frames = list(frame_generator(pcm_data, sample_rate, frame_duration))
    voiced_segments = []
    padding_frames = int(padding_ms / frame_duration)
    ring_buffer = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced = b''

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append(frame)
            if sum(vad.is_speech(f, sample_rate) for f in ring_buffer) > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced += b''.join(ring_buffer)
                ring_buffer.clear()
        else:
            voiced += frame
            ring_buffer.append(frame)
            if sum(not vad.is_speech(f, sample_rate) for f in ring_buffer) > 0.9 * ring_buffer.maxlen:
                triggered = False
                voiced_segments.append(voiced)
                voiced = b''
                ring_buffer.clear()
    if voiced:
        voiced_segments.append(voiced)

    if voiced_segments:
        return AudioSegment(
            data=b''.join(voiced_segments),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
    else:
        return None

def generate_longform_audio(dataset_path: str, vad_config: dict) -> list:
    tsv_file = [f for f in os.listdir(dataset_path) if f.endswith('.tsv')]
    if not tsv_file:
        raise FileNotFoundError("No TSV file found")
    tsv_path = os.path.join(dataset_path, tsv_file[0])
    df = pd.read_csv(tsv_path, sep='\t')

    group_by = vad_config.get("group_by")
    if group_by not in df.columns:
        raise ValueError(f"'{group_by}' column not found")

    silence_threshold = vad_config.get("silence_threshold", 0.5)
    padding = vad_config.get("padding", 0.2)
    overlap = vad_config.get("overlap", 0.1)
    min_duration = vad_config.get("min_duration", 0.0)

    vad = webrtcvad.Vad()
    vad.set_mode(3)

    os.makedirs("longform_outputs", exist_ok=True)
    metadata = []
    output_paths = []

    grouped = df.groupby(group_by)
    for group_id, items in grouped:
        long_audio = AudioSegment.empty()
        segments = []
        for _, row in items.iterrows():
            clip_path = os.path.join(dataset_path, row['path'])
            if not os.path.exists(clip_path):
                logging.warning(f"Missing file: {clip_path}")
                continue

            trimmed = vad_trim(clip_path, vad)
            if trimmed is None:
                logging.info(f"Silent: {clip_path}")
                continue

            if trimmed.duration_seconds < min_duration:
                logging.info(f"Skipped short clip: {clip_path}")
                continue

            padded = AudioSegment.silent(duration=padding * 1000) + trimmed + AudioSegment.silent(duration=padding * 1000)
            if len(long_audio) > 0:
                long_audio = long_audio[:-int(overlap * 1000)]
            start = len(long_audio)
            long_audio += padded
            end = len(long_audio)
            segments.append({
                "source": clip_path,
                "start_ms": start,
                "end_ms": end,
                group_by: group_id
            })

        if len(long_audio) == 0:
            continue

        output_path = f"longform_outputs/{group_by}_{group_id}.wav"
        long_audio.export(output_path, format="wav")
        output_paths.append(output_path)
        metadata.extend(segments)

    with open("longform_outputs/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Processed groups: {len(output_paths)}, Metadata entries: {len(metadata)}")
    return output_paths