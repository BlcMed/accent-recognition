import matplotlib.pyplot as plt
import librosa

# Plot the audio waveform
def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

def plot_spectrogram(spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()

def plot_mel_spectrogram(mel_spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, x_axis="time", y_axis="mel")
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.show()

def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")
    plt.show()