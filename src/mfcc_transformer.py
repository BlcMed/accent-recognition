import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from extract_features import compute_mfcc

class MfccTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variables: List[str],
        sampling_rating, 
        n_mfcc,
        duration,
        overlap
    ):
        self.variables = variables
        self.sampling_rating=sampling_rating, 
        self.n_mfcc=n_mfcc,
        self.duration=duration,
        self.overlap=overlap

    def fit(self, X: pd.DataFrame, y=None):
        audio_data = X["audio"]
        self.mfccs = []
        for audio in audio_data:
            mfcc = compute_mfcc(
                audio,
                self.sampling_rating, 
                self.n_mfcc, 
                self.duration, 
                self.overlap
            )
        self.mfccs.append(mfcc)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X=X.copy()
        X["mfcc"]=self.mfccs
        return X
