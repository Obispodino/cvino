import pandas as pd
from cv_functions.model import load_model
from cv_functions.encoder import Encoder_features_transform
from cv_functions.geocode_regions import retrieve_coordinate
import ipdb
import os
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


LOCAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
pickle_file = os.path.join(LOCAL_DATA_PATH, "trained_model_Dino.pkl")

def get_wine_recommendations_by_characteristics(df,
    wine_type='Red',                   # Wine type (Red, White, Rosé, Sparkling, Dessert, Dessert/Port etc.)
    grape_varieties=None,          # Grape varieties (single string or list)
    body='Full-bodied',              #['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    abv=12.0,                          # Alcohol by volume percentage, min 0, max 50
    acidity=None,                      # Optional - Acidity (Low, Medium, High)
    country=None,                      # Optional - Country of origin
    region_name=None,
    n_recommendations=5, # Number of recommendations to return
    load_model_file=pickle_file
):


    # if there's an entry in region_name
    latitude_in = 0
    longitude_in = 0
    if region_name is not None:
        latitude, longitude = retrieve_coordinate(region_name) # get coordinate from cache
        if not np.isnan(latitude):
            latitude_in = latitude
            longitude_in = longitude

    # create a DataFrame
    X_pred = pd.DataFrame(dict(
    Type = [wine_type],
    ABV=[abv],
    Body=[body],
    Acidity=[acidity],
    Country=[country],
    RegionName=[region_name],
    latitude = [latitude_in],
    longitude = [longitude_in],
    Grapes_list = [grape_varieties],
    avg_rating = [3.79],
    rating_count = [0],
    rating_std = [0]
    ))
    #ipdb.set_trace()
    # turn all None into np.nan so it can be inputed to default values
    X_pred_cleaned = X_pred.replace({None: np.nan})

    # transfor an dencode the incoming data
    wine_proccessed = Encoder_features_transform(X_pred_cleaned)
    # load knn model
    knn_model = load_model(load_model_file)

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

    result = recommended_wines
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


## this is the main function to run the script and test the recommendation funciton
if __name__ == "__main__":
    # Define correct path to raw_data/wine_lookup.csv
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw_data", "wine_lookup.csv"))
    wine_scaled_df = pd.read_csv(data_path)

    LOCAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    pickle_file = os.path.join(LOCAL_DATA_PATH, "trained_model_Dino.pkl")

    recommended_wines = get_wine_recommendations_by_characteristics(wine_scaled_df,
    wine_type='Red',                # Wine type (Red, White, Rosé, Sparkling, etc.)
    grape_varieties=['Cabernet Sauvignon'],       # Grape varieties (single string or list)
    body='Full-bodied',           # Body ['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    acidity='Low',               # Acidity (Low, Medium, High)
    abv=13.5,                       # Alcohol by volume percentage
    country='Chile',               # Country of origin
    region_name='Maule Valley',  # Region name
    n_recommendations=5,
    load_model_file=pickle_file)
    #ipdb.set_trace()

    if not recommended_wines.empty:
        print("Recommended wines based on your preferences:")
        for i, (_, wine) in enumerate(recommended_wines.iterrows(), 1):
            grapes_list = ast.literal_eval(wine['Grapes_list'])
            grapes_str = ', '.join(grapes_list) if isinstance(grapes_list, list) else 'Unknown'
            print(f"{i}. {wine['WineName']} ({wine['Country']}, {wine['Type']})")
            print(f"   Region: {wine['RegionName'] or 'Unknown'}")
            print(f"   Grapes: {grapes_str}")
            print(f"   Body: {wine['Body'] or 'Unknown'}, Acidity: {wine['Acidity'] or 'Unknown'}")
            print(f"   Rating: {wine['avg_rating']:.1f}/5")
            print(f"   Similarity: {wine['Similarity']:.2f}")
            print()
    else:
        print("No recommendations found. Try adjusting your preferences.")
