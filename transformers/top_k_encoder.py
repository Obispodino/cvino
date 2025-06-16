import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import ipdb

class TopNGrapeOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=60, output_prefix='Grape'):
        self.top_n = top_n
        self.output_prefix = output_prefix
        self.top_grapes = []
        self.output_columns = []
        self._transform_output = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        all_grapes = [
            grape
            for sublist in X
            if isinstance(sublist, list)
            for grape in sublist
        ]

        self.top_grapes = pd.Series(all_grapes).value_counts().head(self.top_n).index.tolist()
        self.output_columns = [f'{self.output_prefix}_{grape}' for grape in self.top_grapes]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        def parse(grapes):
            if isinstance(grapes, list):
                return grapes
            elif isinstance(grapes, str):
                return [g.strip() for g in grapes.split(',')]
            return []

        grape_lists = X.apply(parse)

        data = []

        for grapes in grape_lists:
            row = [1 if grape in grapes else 0 for grape in self.top_grapes]
            data.append(row)

        df_output = pd.DataFrame(data, columns=self.output_columns, index=X.index)

        if self._transform_output == 'pandas':
            return df_output
        else:
            return df_output.to_numpy()

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return self.output_columns
