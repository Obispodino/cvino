import pandas as pd
import numpy as np
import ast
from collections import Counter

def get_wine_recommendations_by_food(
    features_df,                     # ✅ Added features_df as parameter
    food_pairing,                    # Required: Food you want to pair with (e.g., "steak", "pasta")
    wine_type=None,                  # Optional: Wine type (Red, White, Rosé, Sparkling, etc.)
    grape_varieties=None,            # Optional: Grape varieties (single string or list)
    body=None,                       # Optional: Body (Very light to Very full-bodied)
    abv=None,                        # Optional: Alcohol by volume percentage
    acidity=None,                    # Optional: Acidity (Low, Medium, High)
    country=None,                    # Optional: Country of origin
    region_name=None,                # Optional: Region name
    n_recommendations=5,             # Number of recommendations to return
    exact_match_only=False           # If True, only return wines with exact food match
):
    """
    Get wine recommendations based on food pairing and optional wine characteristics.
    This function performs a reverse lookup — it starts with the desired food
    and finds wines that pair well with it.
    """
    # Step 1: Prepare food pairing search term
    food_search = food_pairing.lower().strip()

    # Step 2: Ensure Harmonize is properly processed as a list
    def safe_eval_list(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return x if isinstance(x, list) else []

    working_df = features_df.copy()
    working_df['Harmonize'] = working_df['Harmonize'].apply(safe_eval_list)

    # Step 3: Define function to check if a food is in the harmonize list
    def contains_food(harmonize_list, food_term):
        if not isinstance(harmonize_list, list):
            return False
        harmonize_lower = [item.lower() for item in harmonize_list]
        if food_term in harmonize_lower:
            return True
        if not exact_match_only:
            return any(food_term in item for item in harmonize_lower)
        return False

    # Step 4: Create food match column
    working_df['food_match'] = working_df['Harmonize'].apply(
        lambda x: contains_food(x, food_search))

    food_matched_wines = working_df[working_df['food_match']]

    if food_matched_wines.empty:
        print(f"No wines found that pair with '{food_pairing}'. Try a different food.")
        return pd.DataFrame()

    # Step 5: Apply additional filters
    if wine_type is not None:
        food_matched_wines = food_matched_wines[food_matched_wines['Type'] == wine_type]

    if grape_varieties is not None:
        if isinstance(grape_varieties, str):
            grape_list = [g.strip() for g in grape_varieties.split(',')] if ',' in grape_varieties else [grape_varieties]
        else:
            grape_list = grape_varieties

        def has_any_grape(wine_grapes, target_grapes):
            if not isinstance(wine_grapes, list):
                return False
            return any(grape in wine_grapes for grape in target_grapes)

        food_matched_wines['Grapes_list'] = food_matched_wines['Grapes_list'].apply(safe_eval_list)
        food_matched_wines = food_matched_wines[
            food_matched_wines['Grapes_list'].apply(lambda x: has_any_grape(x, grape_list))]

    if body is not None:
        food_matched_wines = food_matched_wines[food_matched_wines['Body'] == body]

    if country is not None:
        food_matched_wines = food_matched_wines[food_matched_wines['Country'] == country]

    if acidity is not None:
        food_matched_wines = food_matched_wines[food_matched_wines['Acidity'] == acidity]

    if region_name is not None:
        food_matched_wines = food_matched_wines[food_matched_wines['RegionName'] == region_name]

    if food_matched_wines.empty:
        print(f"No wines found that match all your criteria with '{food_pairing}'.")
        return pd.DataFrame()

    # Step 6: Sort and return results
    result = food_matched_wines.sort_values('avg_rating', ascending=False).head(n_recommendations)
    return result
