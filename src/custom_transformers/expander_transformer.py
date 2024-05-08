import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class ExpanderTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, variables_to_repeat, n_mfcc):
        columns_to_expand = []
        for i in range(n_mfcc):
            columns_to_expand.append(f"mfcc_{i+1}")
        self.columns_to_expand = columns_to_expand
        self.variables_to_repeat = variables_to_repeat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X=X.copy()
        columns = self.variables_to_repeat + self.columns_to_expand
        X_expanded=pd.DataFrame(columns = columns)
        columns_to_repeat = self.variables_to_repeat
        for index, row in X.iterrows():
            dict_repeated={}
            print(f'===================== \n We are in iteration {index}')
            for col in columns_to_repeat:
                dict_repeated[col]=row[col]
            expanding_length = len(row[self.columns_to_expand[0]])
            for i in range(expanding_length):
                row_to_append={}
                for col in self.columns_to_expand:
                    row_to_append[col] = [row[col][i]]
                row_to_append.update(dict_repeated)
                pd_row = pd.DataFrame(row_to_append)
                X_expanded = pd.concat([X_expanded, pd_row])
        return X_expanded


if __name__=="__main__":
    data={
        "A":['label','ad'],
        "B":['lbl','t'],
        "C":['dffdfbl','eeee'],
        "mfcc_1":[np.array([2,4,3]),np.array([4,7,2])],
        "mfcc_2":[np.array([2,2,3]),np.array([1,1,5])]
    }
    import pandas as pd
    df=pd.DataFrame(data=data)
    expander=ExpanderTransformer(["A","C"],n_mfcc=2)
    df_tr=expander.fit_transform(df)
    print("----------------")
    print(df_tr)
