import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables= variables
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def trasnform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self