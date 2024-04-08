import numpy as np
import librosa


frame_lenght = 2048
hop_length = 512
n_mfcc = 13


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


def compute_mfcc(audio, sr, n_mfcc=n_mfcc):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def compute_LPC():
    return 0


def compute_PLP():
    return 0
