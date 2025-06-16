from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

class BodyOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, categories, fallback='Medium-bodied', output_name='Body_encoded'):
        self.column = column
        self.categories = categories  # must be list of lists for OrdinalEncoder
        self.fallback = fallback
        self.output_name = output_name
        self._transform_output = None  # to track output format

    def fit(self, X, y=None):
        categories_copy = [cat_list[:] for cat_list in self.categories]

        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].where(
            X_copy[self.column].isin(categories_copy[0]), self.fallback
        )

        self.encoder_ = OrdinalEncoder(categories=categories_copy)
        self.encoder_.fit(X_copy[[self.column]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].where(
            X_copy[self.column].isin(self.categories), self.fallback
        )

        encoded = self.encoder_.transform(X_copy[[self.column]])

        if self._transform_output == 'pandas':
            return pd.DataFrame(encoded, columns=[self.output_name], index=X.index)
        else:
            return encoded  # returns a NumPy array

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self
