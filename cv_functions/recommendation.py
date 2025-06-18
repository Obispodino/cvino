import pandas as pd
from cv_functions.model import load_model
from cv_functions.encoder import Encoder_features_transform
from cv_functions.geocode_regions import retrieve_coordinate
import numpy as np
import os
import ast
from fastapi import HTTPException

# # Load model path
# LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_model.pkl"))
# model = load_model(LOCAL_MODEL_PATH)

def get_wine_recommendations_by_characteristics(
    wine_type='Red',
    grape_varieties=None,
    body='Full-bodied',
    abv=12.0,
    acidity=None,
    country=None,
    region_name=None,
    n_recommendations=5,
    metadata_df: pd.DataFrame = None,
    model=None
):
    latitude, longitude = 0, 0
    if region_name:
        lat, lon = retrieve_coordinate(region_name)
        if not np.isnan(lat):
            latitude, longitude = lat, lon

    X_pred = pd.DataFrame([{
        "Type": wine_type,
        "ABV": abv,
        "Body": body,
        "Acidity": acidity,
        "Country": country,
        "RegionName": region_name,
        "latitude": latitude,
        "longitude": longitude,
        "Grapes_list": grape_varieties,
        "avg_rating": 3.79,
        "rating_count": 0,
        "rating_std": 0
    }])

    X_pred_cleaned = X_pred.replace({None: np.nan})
    wine_processed = Encoder_features_transform(X_pred_cleaned)
    distances, indices = model.kneighbors(wine_processed, n_neighbors=max(n_recommendations * 3, 20))

    recommended_ids = metadata_df.iloc[indices[0]]["WineID"].tolist()
    distance_map = {wid: dist for wid, dist in zip(recommended_ids, distances[0])}

    recommended = metadata_df[metadata_df["WineID"].isin(recommended_ids)].copy()
    recommended["Similarity"] = recommended["WineID"].map(lambda x: 1 - distance_map.get(x, 1))

    if country:
        filtered = recommended[recommended["Country"] == country]
        if not filtered.empty:
            recommended = filtered

    return recommended.sort_values("Similarity", ascending=False).head(n_recommendations)
