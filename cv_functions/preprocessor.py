from cv_functions.encoder import Encoder_features_transform
from transformers.ratings_stat import RatingsStatsAggregator


def preprocess_features(df):

    wine_proccessed = Encoder_features_transform(df)
    ratings_stats = RatingsStatsAggregator.transform(df)
