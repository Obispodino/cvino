import pickle
import os
import pandas as pd
from pathlib import Path
import ipdb

from cv_functions.recommendation import get_wine_recommendations_by_characteristics
from cv_functions.geocode_regions import geocode_regions
from cv_functions.preprocessor import wine_clean_features, ratings_clean_features, preprocess_features
from cv_functions.model import train_model

# load data
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data", "last")
wines_file = os.path.join(LOCAL_DATA_PATH, "XWines_Full_100K_wines.csv")
wines_data = pd.read_csv(wines_file)

ratings_file = os.path.join(LOCAL_DATA_PATH, "XWines_Full_21M_ratings.csv")
ratings_data = pd.read_csv(ratings_file)

# determine parameters
N_TOP_GRAPES = 50

# step1 : geocode the region
wd = geocode_regions(wines_data, min_delay=2.0)

# step2 : clean_features (from preprocessor)
wines_clean_df = wine_clean_features(wd)
ratings_clean_df = ratings_clean_features(ratings_data, wines_clean_df)

# step3 : preprocess_features (features_df, ratings_clean, N_TOP_GRAPES) (from preprocessor)
X_scaled_df = preprocess_features(wines_clean_df, ratings_clean_df, N_TOP_GRAPES = 50)

# step4 : train model and return model (from model)
knn_model = train_model(X_scaled_df, n_neighbors = 6)

def get_wine_recommendations(wine_id, n_recommendations=5):
    """Get wine recommendations based on k-NN similarity"""
    try:
        # Find the index of the wine
        wine_idx = wines_clean_df[wines_clean_df['WineID'] == wine_id].index[0]

        # Get the wine's feature vector
        wine_features = X_scaled_df.loc[wine_idx].values.reshape(1, -1)

        # Find similar wines
        distances, indices = knn_model.kneighbors(wine_features, n_neighbors=n_recommendations+1)

        # Get recommended wine IDs (excluding the input wine itself)
        recommended_indices = indices[0][1:]  # Skip the first one (itself)
        recommended_wine_ids = wines_clean_df.iloc[recommended_indices]['WineID'].tolist()

        return recommended_wine_ids, distances[0][1:]
    except IndexError:
        return [], []


# Example usage Input here
recommendations = get_wine_recommendations_by_characteristics(
    wine_type='Red',
    grape_varieties='Pinot Noir, Cabernet Sauvignon',  # Can be a single string or a list
    body='Medium-bodied', #['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    acidity='Low',
    abv=12.0,
    country=None,
    region_name='Margaret River',  # Example region
    n_recommendations=14
)

# Display recommendations
print("Recommended wines based on your preferences:")
for i, (_, wine) in enumerate(recommendations.iterrows(), 1):
    grapes_str = ', '.join(wine['Grapes_list']) if isinstance(wine['Grapes_list'], list) else 'Unknown'
    print(f"{i}. {wine['WineName']} ({wine['Country']}, {wine['Type']})")
    print(f"   Grapes: {grapes_str}")
    print(f"   Body: {wine['Body'] or 'Unknown'}, Acidity: {wine['Acidity'] or 'Unknown'}")
    print(f"   Rating: {wine['avg_rating']:.1f}/5")
    print(f"   Similarity: {wine['Similarity']:.2f}")
    print()
