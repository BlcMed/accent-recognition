import numpy as np
import librosa


def compute_spectrogram(audio):
    # Compute Short Time Fourier Transform
    D = librosa.stft(y=audio)
    # Convert to decibels
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return spectrogram


def compute_mel_spectrogram(audio, sr):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmax=8000
    )
    return mel_spectrogram


def compute_mfcc(
    audio, sr, n_mfcc, duration=0.025, overlap=0.010
):  # duration=25ms, overlap=10ms

    # Convert duration and overlap from seconds to samples
    frame_length = int(sr * duration)
    hop_length = int(sr * overlap)

    # Calculate n_fft (next power of 2 of frame_length) for FFT to ensure efficient FFT computation
    n_fft = 2 ** (frame_length - 1).bit_length()

    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc


def compute_LPC():
    return 0


def compute_PLP():
    return 0
