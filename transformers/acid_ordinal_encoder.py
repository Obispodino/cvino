from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

class AcidOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, categories, fallback='Medium',output_name='Acidity_encoded'):
        self.column = column
        self.categories = categories  # must be list of lists for OrdinalEncoder
        self.fallback = fallback
        self.output_name = output_name

    def fit(self, X, y=None):
        # Use a copy of categories to avoid modifying the original parameter
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
        return pd.DataFrame(encoded, columns=[self.output_name], index=X.index)
