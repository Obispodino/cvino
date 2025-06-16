import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import ipdb

# Import from transformers/top_k_encoder.py
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

        all_grapes = [grape for sublist in X if isinstance(sublist, list) for grape in sublist]

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

# Import from transformers/body_ordinal_encoder.py
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

# Import from transformers/acid_ordinal_encoder.py
class AcidOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column, categories, fallback='Medium', output_name='Acidity_encoded'):
        self.column = column
        self.categories = categories  # must be list of lists for OrdinalEncoder
        self.fallback = fallback
        self.output_name = output_name
        self._transform_output = None  # to track output format

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

        if self._transform_output == 'pandas':
            return pd.DataFrame(encoded, columns=[self.output_name], index=X.index)
        else:
            return encoded  # returns a NumPy array

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

# Import from transformers/ratings_stat.py
class RatingsStatsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, ratings_df, wine_id_col='WineID', rating_col='Rating', scale=True):
        self.ratings_df = ratings_df
        self.wine_id_col = wine_id_col
        self.rating_col = rating_col
        self.scale = scale
        self.ratings_stats_ = None
        self.scaler = MinMaxScaler() if scale else None
        self._transform_output = None

    def fit(self, X=None, y=None):
        # Compute aggregated rating statistics
        stats = (
            self.ratings_df.groupby(self.wine_id_col, as_index=True)
            .agg({self.rating_col: ['mean', 'count', 'std']})
            .round(2)
        )
        stats.columns = ['avg_rating', 'rating_count', 'rating_std']
        self.output_columns = ['avg_rating', 'rating_count', 'rating_std']
        stats = stats.fillna(0)

        # Save unscaled stats for merging
        self.ratings_stats_ = stats

        # Fit the scaler if enabled
        if self.scale:
            self.scaler.fit(stats)

        return self

    def transform(self, X):
        if self.ratings_stats_ is None:
            raise RuntimeError("Must call fit() before transform()")

        X = X.copy()
        # Merge stats into input DataFrame
        X = X.merge(self.ratings_stats_, on=self.wine_id_col, how='left')
        X[['avg_rating', 'rating_count', 'rating_std']] = X[['avg_rating', 'rating_count', 'rating_std']].fillna(0)

        # Apply scaling if enabled
        if self.scale:
            scaled = self.scaler.transform(X[['avg_rating', 'rating_count', 'rating_std']])
            X[['avg_rating', 'rating_count', 'rating_std']] = scaled

        return X

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return self.output_columns
