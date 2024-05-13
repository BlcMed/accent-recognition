from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_mfccs):
        self.variables = [f"mfcc_{i+1}" for i in range(n_mfccs)]
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.variables])
        return self
    
    def transform(self, X):
        X[self.variables] = self.scaler.transform(X[self.variables])
        return X 

