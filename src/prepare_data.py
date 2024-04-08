import os
import librosa
from extract_features import compute_mfcc


# load Audio files_with labels
def load_audio_files(folder_path, sr):
    audio_files = os.listdir(folder_path)
    audio_data = []
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        waveform, sr = librosa.load(file_path, sr=sr)
        # Extract label
        label = file.split(".")[0]
        label = "".join([i for i in label if not i.isdigit()])
        audio_data.append([waveform, label])
    return audio_data


def split_audio_by_silence(
    audio, sr, threshold_percentage=0.01, min_silence_duration=1
):

    y, label = audio
    # Compute energy of audio segments
    energy = librosa.feature.rms(y=y)[0]

    # Find segments below energy threshold
    max_energy = max(energy)
    threshold_energy = threshold_percentage * max_energy
    silent_segments = energy < threshold_energy

    # Identify split points
    split_points = []
    start_point = None
    for i, segment in enumerate(silent_segments):
        if segment:
            if start_point is None:
                start_point = i
        elif start_point is not None:
            if (i - start_point) >= (min_silence_duration * sr):
                split_points.append((start_point, i))
            start_point = None

    # Split audio file at identified points
    audio_segments = []
    prev_end = 0
    for start, end in split_points:
        waveform_segment = y[start * sr : end * sr]
        audio_segments.append((waveform_segment, label))
        prev_end = end * sr
    audio_segments.append(y[prev_end:])

    return audio_segments


# Segment Audio into smaller chunks
def segment_audio(
    waveform, sr, duration=0.01, overlap=0.001
):  # duration=10ms, overlap=1ms

    # Step 1: Compute the number of samples per segment
    samples_per_segment = int(sr * duration)
    # Step 2: Compute the number of samples to overlap
    samples_per_overlap = int(samples_per_segment * overlap)
    # Step 3: Compute the total number of segments
    total_segments = int(len(waveform) / (samples_per_segment - samples_per_overlap))
    # Step 4: Create an empty list to store segments
    segments = []
    # Step 5: Create a loop to extract segments
    for i in range(total_segments):
        start = samples_per_segment * i
        end = start + samples_per_segment
        segment = waveform[start:end]
        segments.append(segment)
    return segments
