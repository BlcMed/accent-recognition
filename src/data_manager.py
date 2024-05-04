import os
import librosa
import pandas as pd

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
        df = pd.DataFrame({'audio': audio_data, 'labels': labels})
    return df
    #return audio_data, labels


def filter_data_based_on_accents(df, considered_accents):
    audio_data=df["audio"]
    labels=df["labels"]
    audio_data_filtered=[]
    labels_filtered=[]
    for i,audio in enumerate(audio_data):
        if labels[i] in considered_accents:
            audio_data_filtered.append(audio)
            labels_filtered.append(labels[i])
    df = pd.DataFrame({'audio': audio_data_filtered, 'labels': labels_filtered})
    return df


'''
def filter_data_based_on_accents(audio_data, labels, considered_accents):
    audio_data_filtered=[]
    labels_filtered=[]
    for i,audio in enumerate(audio_data):
        if labels[i] in considered_accents:
            audio_data_filtered.append(audio)
            labels_filtered.append(labels[i])
    return audio_data_filtered, labels_filtered'''