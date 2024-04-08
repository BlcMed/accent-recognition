import os
import librosa
from extract_features import compute_mfcc
import numpy as np


# load Audio files_with labels
def load_audio_files(folder_path, sr):
    audio_files = os.listdir(folder_path)
    audio_data = []
    labels = []
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=sr)
        audio_data.append(audio)
        # Extract label
        label = file.split(".")[0]
        label = "".join([i for i in label if not i.isdigit()])
        labels.append(label)
    return audio_data, labels


def split_audio_by_silence(
    audio, sr, threshold_percentage=0.01, min_silence_duration=1
):
    # Compute energy of audio frames
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    # Find frames below energy threshold
    max_energy = max(energy)
    threshold_energy = threshold_percentage * max_energy
    silent_frames = energy < threshold_energy  # logical array
    # Identify split points
    split_points = []  # for silent parts
    start_point = None
    min_silence_samples = min_silence_duration * sr

    for i, silent_frame in enumerate(silent_frames):
        current_sample = i * hop_length
        if silent_frame:
            if start_point is None:
                start_point = current_sample
        elif start_point is not None:
            if (current_sample - start_point) >= min_silence_samples:
                split_points.append((start_point, current_sample))
            start_point = None

    # Trim audio2 file at identified points (unwanted silence) and return audible segments
    audible_segments = []

    # Append the initial segment if not silent
    if len(split_points) > 0 and split_points[0][0] != 0:
        audible_segments.append(audio[: split_points[0][0]])

    # Append non-silent segments
    for i in range(len(split_points) - 1):
        start_idx = split_points[i][1]
        end_idx = split_points[i + 1][0]
        if start_idx != end_idx:
            audible_segments.append(audio[start_idx:end_idx])

    # Append the last segment if not silent
    if len(split_points) > 0 and split_points[-1][1] != len(audio):
        audible_segments.append(audio[split_points[-1][1] :])

    # Return audible segments list of arrays
    # each array contains the audio samples of a segment of the original audio file where sound is present
    return audible_segments


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
