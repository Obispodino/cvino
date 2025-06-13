import pandas as pd
from cv_functions.model import load_model
from cv_functions.encoder import Encoder_features_transform
from cv_functions.geocode_regions import retrieve_coordinate
import ipdb
import os
import numpy as np


def get_wine_recommendations_by_characteristics(df,
    wine_type='Red',                # Wine type (Red, White, Rosé, Sparkling, etc.)
    grape_varieties='Syrah/Shiraz',       # Grape varieties (single string or list)
    body='Full-bodied',           # Body ['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    acidity='Medium',               # Acidity (Low, Medium, High)
    abv=14.0,                       # Alcohol by volume percentage
    country='Australia',               # Country of origin
    region_name='Maule Valley',  # Region name
    n_recommendations=5             # Number of recommendations to return
):


    # if there's an entry in region_name
    if region_name is not None:
        latitude, longitude = retrieve_coordinate(region_name) # get coordinate from cache
        if not np.isnan(latitude):
            latitude_in = latitude
            longitude_in = longitude
    else:
        latitude_in = [42.965916],
        longitude_in = [2.8390454],


    # create a DataFrame
    X_pred = pd.DataFrame(dict(
    Type =wine_type,
    ABV=abv,
    Body=body,
    Acidity=acidity,
    Country=country,
    RegionName=region_name,
    latitude = latitude_in,
    longitude = longitude_in,
    Grapes_list = grape_varieties,
    avg_rating = [3.79],
    rating_count = [0],
    rating_std = [0]
    ))
    ipdb.set_trace()
    wine_proccessed = Encoder_features_transform(X_pred)
    knn_model = load_model()

    # Get extra recommendations to account for filtering
    n_neighbors = max(n_recommendations * 3, 20) if country else n_recommendations

    distances, indices = knn_model.kneighbors(wine_proccessed, n_neighbors=n_neighbors)

    # Retrieve wine IDs
    recommended_indices = indices[0]
    recommended_wine_ids = df.iloc[recommended_indices]['WineID'].tolist()

    # Get full wine details
    recommended_wines = df[df['WineID'].isin(recommended_wine_ids)][
    ['WineID', 'WineName', 'Country', 'Type', 'RegionName', 'Body', 'Acidity', 'Grapes_list', 'avg_rating']
    ]

    # Add similarity scores
    distance_map = {wine_id: dist for wine_id, dist in zip(recommended_wine_ids, distances[0])}
    recommended_wines['Similarity'] = recommended_wines['WineID'].map(lambda wine_id: 1 - distance_map[wine_id])

    # Filter by country if specified
    if country:
        country_matches = recommended_wines[recommended_wines['Country'] == country]
        if not country_matches.empty:
            recommended_wines = country_matches
        else:
            print(f"Warning: No wines found from '{country}'. Showing wines from all countries.")

        # Sort by similarity and limit results
        result = recommended_wines.sort_values('Similarity', ascending=False).head(n_recommendations)

    if result.empty:
        print("No matching wines found. Try different criteria.")
        return pd.DataFrame()

    return result

if __name__ == "__main__":
    LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    wine_scaled_df = pd.read_csv(os.path.join(LOCAL_DATA_PATH, 'wine_lookup.csv'))

    recommended_wines = get_wine_recommendations_by_characteristics(wine_scaled_df,     wine_type='Red',                # Wine type (Red, White, Rosé, Sparkling, etc.)
    grape_varieties='Syrah/Shiraz',       # Grape varieties (single string or list)
    body='Full-bodied',           # Body ['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    acidity='Medium',               # Acidity (Low, Medium, High)
    abv=14.0,                       # Alcohol by volume percentage
    country='Australia',               # Country of origin
    region_name='Maule Valley',  # Region name
    n_recommendations=5  )

    recommended_wines
