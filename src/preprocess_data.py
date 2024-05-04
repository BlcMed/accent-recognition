import os
import librosa


def split_audio_by_silence(
    audio,
    sr,
    threshold_percentage=0.01,
    min_silence_duration=1,
    frame_length_energy=2048,
    hop_length=512,
):
    # Compute energy of audio frames
    energy = librosa.feature.rms(
        y=audio, frame_length=frame_length_energy, hop_length=hop_length
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

    # Append the whole audio if no silent parts
    if len(split_points) == 0:
        audible_segments.append(audio)

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
    # Each array contains the audio samples of a segment of the original audio file where sound is present
    return audible_segments


def split_all_audios_by_silence(audio_data, labels, sampling_rating, threshold_percentage, min_silence_duration):
    audio_data_trimmed = []
    labels_trimmed = []
    for i, audio in enumerate(audio_data):
        audible_segments = split_audio_by_silence(
            audio,
            sampling_rating,
            threshold_percentage=threshold_percentage,
            min_silence_duration=min_silence_duration,
        )
        audio_data_trimmed.extend(audible_segments)
        labels_trimmed.extend([labels[i]] * len(audible_segments))
    return audio_data_trimmed, labels_trimmed

# Segment Audio into smaller chunks
def segment_audio(
    audio, sr, duration=0.025, overlap=0.010
):  # duration=25ms, overlap=10ms

    # Convert duration and overlap from seconds to samples
    samples_per_segment = int(sr * duration)
    hop_length = int(sr * overlap)

    # Compute the total number of segments
    total_segments = 1 + (len(audio) - samples_per_segment) // hop_length

    # Initialize an empty list to store segments
    segments = []

    # Create a loop to extract segments
    for i in range(total_segments):
        start = samples_per_segment * i
        end = start + samples_per_segment
        segment = audio[start:end]
        segments.append(segment)
    return segments

