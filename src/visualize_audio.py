import matplotlib.pyplot as plt
import librosa


def plot_audio(waveform, sr, title="Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y=waveform, sr=sr, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()


def plot_spectrogram(spectrogram, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, x_axis="time", y_axis="mel")
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_mfcc(mfcc, title="MFCC"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.show()
