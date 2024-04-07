import numpy as np
import librosa


def load_audio(audio_path):
    y, sr = librosa.load(audio_path)
    return y, sr


def compute_spectrogram(y):
    D = librosa.stft(y)
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return spectrogram


def compute_mel_spectrogram(y, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spectrogram


def compute_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc


def compute_LPC():
    return 0


def compute_PLP():
    return 0


# usage example
y, sr = load_audio("./data/raw/english1.mp3")

spectrogram = compute_spectrogram(y)
mel_spectrogram = compute_mel_spectrogram(y, sr)
mfcc = compute_mfcc(y, sr)

import visualize_audio as va

# va.plot_waveform(y, sr)
# va.plot_spectrogram(spectrogram)
# va.plot_mel_spectrogram(mel_spectrogram)
# va.plot_mfcc(mfcc)
