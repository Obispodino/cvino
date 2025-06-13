from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
            self.ratings_df.groupby(self.wine_id_col)
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
        X = X.merge(self.ratings_stats_, left_on=self.wine_id_col, right_index=True, how='left')
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
