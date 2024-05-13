from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        self.variable = variable
        self.label_encoder = LabelEncoder()
    
    def fit(self, X, y=None):
        self.label_encoder.fit(X[self.variable])
        return self
    
    def transform(self, X):
        X[self.variable] = self.label_encoder.transform(X[self.variable])
        return X

