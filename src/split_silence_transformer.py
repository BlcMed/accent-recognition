import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from preprocess_data import split_audio_by_silence


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


class SplitSilenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables= variables
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def trasnform(self, X: pd.DataFrame) -> pd.DataFrame:
        audio_data, labels = X["audio"]
        audio_data_trimmed, labels_trimmed = split_all_audios_by_silence(
            audio_data=audio_data,
            labels=labels)
        df = pd.DataFrame({'audio': audio_data, 'Labels': labels})
        return df