import os
import librosa
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline


def load_audio_files(folder_path, sr):
    audio_files = os.listdir(folder_path)
    audio_data = []
    labels = []
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=sr)
        audio_data.append(audio)
        label = extract_label(file)
        labels.append(label) 
        df = pd.DataFrame({'audio': audio_data, 'label': labels})
    return df


def extract_label(file):
    # Remove the .mp3 extention
    label = file.split(".")[0]
    # Remove the number
    label = "".join([i for i in label if not i.isdigit()])
    return label


def filter_data_based_on_accents(df, considered_accents):
    audio_data=df["audio"]
    labels=df["label"]
    audio_data_filtered=[]
    labels_filtered=[]
    for i,audio in enumerate(audio_data):
        if labels[i] in considered_accents:
            audio_data_filtered.append(audio)
            labels_filtered.append(labels[i])
    df = pd.DataFrame({'audio': audio_data_filtered, 'label': labels_filtered})
    return df


def save_pipeline(*, pipeline: Pipeline):
    save_file_name = ".pkl"
    save_path = " " / save_file_name
    joblib.dump(pipeline, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = " "/ file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
