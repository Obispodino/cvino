import pandas as pd

def Rates_aggregator(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates ratings statistics for wines and merges them with wine features.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing wine ratings.

    Returns:
        pd.DataFrame: DataFrame with aggregated ratings statistics merged with wine features.
    """
    # Ensure the WineID column is present in both DataFrames
    if 'WineID' not in ratings_df.columns or 'WineID' not in ratings_df.columns:
        raise ValueError("Both ratings_df must contain 'WineID' column.")

    # Aggregate ratings statistics
    # Group by WineID and calculate mean, count, and std of ratings

    ratings_clean = ratings_df[ratings_df['Rating'] > 0]  # Filter out non-positive ratings
    if ratings_clean.empty:
        raise ValueError("No valid ratings found after filtering. Ensure ratings are greater than 0.")
    wine_stats = ratings_clean.groupby('WineID', as_index=False).agg({
        'Rating': ['mean', 'count', 'std']
    }).round(2)

    wine_stats.columns = ['WineID','avg_rating', 'rating_count', 'rating_std']
    wine_stats = wine_stats.fillna(0)

    # # Merge with wine features
    # features_df = features_df.merge(wine_stats, left_on='WineID', right_index=True, how='left')
    # features_df[['avg_rating', 'rating_count', 'rating_std']] = features_df[['avg_rating', 'rating_count', 'rating_std']].fillna(0)
    # rating_columns = ['avg_rating', 'rating_count', 'rating_std']

    return wine_stats
