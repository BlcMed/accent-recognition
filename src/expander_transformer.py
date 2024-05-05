import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class ExpanderTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, n_mfcc):
        columns_to_expand = []
        for i in range(n_mfcc):
            columns_to_expand.append(f"mfcc_{i+1}")
        self.columns_to_expand = columns_to_expand

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X=X.copy()
        transformed_data_dict={}
        for col in X.columns:
            transformed_data_dict[col]=[]
        for index, row in X.iterrows():
            expanding_length = len(row[self.columns_to_expand[0]])
            for col in X.columns:
                if col in self.columns_to_expand:
                    row_to_append = row[col].transpose()    
                else:
                    row_to_append = np.repeat(row[col], expanding_length)
                transformed_data_dict[col].extend(row_to_append) 
 
        new_df = pd.DataFrame(transformed_data_dict)
        new_df.columns = X.columns
        return new_df

