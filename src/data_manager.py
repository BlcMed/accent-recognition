import os
import librosa

# Load Audio files_with labels
def load_audio_files(folder_path, sr):
    audio_files = os.listdir(folder_path)
    audio_data = []
    labels = []
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=sr)
        audio_data.append(audio)
        
        # Extract label
        # Remove the .mp3 extention
        label = file.split(".")[0]
        # Remove the number
        label = "".join([i for i in label if not i.isdigit()])
        labels.append(label)
    return audio_data, labels
