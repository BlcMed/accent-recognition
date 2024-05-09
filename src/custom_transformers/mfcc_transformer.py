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
        self.sampling_rating=sampling_rating 
        self.n_mfcc=n_mfcc
        self.duration=duration
        self.overlap=overlap

    def fit(self, X: pd.DataFrame, y=None):
        audio_data = X["audio"]
        self.mfccs = {}
        for i in range(self.n_mfcc):
            self.mfccs[f"mfcc_{i+1}"] = []
        for audio in audio_data:
            mfcc = compute_mfcc(
                audio=audio,
                sampling_rating=self.sampling_rating,
                n_mfcc=self.n_mfcc,
                duration=self.duration,
                overlap=self.overlap
            )
            for i in range(self.n_mfcc):
               self.mfccs[f"mfcc_{i+1}"].append(mfcc[i])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X=X.copy()
        for i in range(self.n_mfcc):
            X[f"mfcc_{i+1}"]=self.mfccs[f"mfcc_{i+1}"]
        return X

if __name__=="__main__":
    print("i am in mfcc transformer")