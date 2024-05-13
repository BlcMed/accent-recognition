import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from preprocess_data import split_audio_by_silence


def split_all_audios_by_silence(audio_data,
                                labels,
                                sampling_rating,
                                threshold_percentage,
                                min_silence_duration,
                                frame_length_energy,
                                hop_length
):
    audio_data_trimmed = []
    labels_trimmed = []
    for i, audio in enumerate(audio_data):
        audible_segments = split_audio_by_silence(
            audio,
            sampling_rating,
            threshold_percentage=threshold_percentage,
            min_silence_duration=min_silence_duration,
            frame_length_energy=frame_length_energy,
            hop_length=hop_length
        )
        audio_data_trimmed.extend(audible_segments)
        labels_trimmed.extend([labels[i]] * len(audible_segments))
    return audio_data_trimmed, labels_trimmed


class SplitSilenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                variables: List[str],
                sampling_rating,
                threshold_percentage,
                min_silence_duration,
                frame_length_energy,
                hop_length
    ):
        self.variables= variables
        self.sampling_rating=sampling_rating
        self.threshold_percentage=threshold_percentage
        self.min_silence_duration=min_silence_duration
        self.frame_length_energy=frame_length_energy
        self.hop_length=hop_length
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X=X.copy()
        audio_data, labels = X["audio"], X["label"]
        audio_data_trimmed, labels_trimmed = split_all_audios_by_silence(
            audio_data=audio_data,
            labels=labels,
            sampling_rating=self.sampling_rating,
            threshold_percentage=self.threshold_percentage,
            min_silence_duration=self.min_silence_duration,
            frame_length_energy=self.frame_length_energy,
            hop_length=self.hop_length
        )
        X = pd.DataFrame({'audio': audio_data_trimmed, 'label': labels_trimmed})
        return X