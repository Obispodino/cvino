import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin

class TopKOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10, prefix='Grape'):
        self.top_k = top_k
        self.prefix = prefix

    def _parse(self, x):
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return []

    def fit(self, X, y=None):
        all_grapes = []
        for val in X.iloc[:, 0]:
            grapes = self._parse(val)
            all_grapes.extend(grapes)
        self.top_grapes_ = pd.Series(all_grapes).value_counts().head(self.top_k).index.tolist()
        return self

    def transform(self, X):
        result = pd.DataFrame(index=X.index)
        for grape in self.top_grapes_:
            col = f"{self.prefix}_{grape.replace(' ', '_')}"
            result[col] = X.iloc[:, 0].apply(lambda x: int(grape in self._parse(x)))
        return result

    def get_feature_names_out(self, input_features=None):
        return [f"{self.prefix}_{g.replace(' ', '_')}" for g in self.top_grapes_]
