import pandas as pd


def get_wine_recommendations_by_characteristics(
    wine_type='Red',                # Wine type (Red, White, Rosé, Sparkling, etc.)
    grape_varieties='Syrah/Shiraz',       # Grape varieties (single string or list)
    body='Full-bodied',           # Body ['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    acidity='Medium',               # Acidity (Low, Medium, High)
    abv=14.0,                       # Alcohol by volume percentage
    country='Australia',               # Country of origin
    region_name='MacLaren Vale',  # Region name
    n_recommendations=5             # Number of recommendations to return
):
    """Get wine recommendations based on specified characteristics."""
    # Initialize feature vector
    input_features = {}

    # Process wine type (one-hot encoding)
    if wine_type in type_encoder.categories_[0]:
        type_vector = type_encoder.transform([[wine_type]])[0]
        for i, col in enumerate(type_columns):
            input_features[col] = type_vector[i]
    else:
        for col in type_columns:
            input_features[col] = 0

    # Process grape varieties (binary features)
    if isinstance(grape_varieties, str):
        grape_varieties = [grape_varieties]

    for col in grape_columns:
        input_features[col] = 0

    for grape in grape_varieties:
        col_name = f'Grape_{grape}'
        if col_name in grape_columns:
            input_features[col_name] = 1

    # Process body (ordinal encoding)
    if body in body_categories:
        body_encoded = body_encoder.transform([[body]])[0][0]
    else:
        body_encoded = 2  # Default to medium-bodied
    input_features['Body_encoded'] = body_encoded

    # Process acidity (ordinal encoding)
    if acidity in acidity_categories:
        acidity_encoded = acidity_encoder.transform([[acidity]])[0][0]
    else:
        acidity_encoded = 1  # Default to medium acidity
    input_features['Acidity_encoded'] = acidity_encoded

    # Process ABV
    input_features['ABV_numeric'] = float(abv)

    # Set default values for remaining features
    input_features['latitude'] = features_df['latitude'].median()
    input_features['longitude'] = features_df['longitude'].median()
    input_features['avg_rating'] = features_df['avg_rating'].median()
    input_features['rating_count'] = 0
    input_features['rating_std'] = 0

    # Create feature DataFrame and get recommendations
    input_df = pd.DataFrame([{col: input_features.get(col, 0) for col in feature_columns}])
    scaled_input = scaler.transform(input_df)
    distances, indices = knn_model.kneighbors(scaled_input, n_neighbors=n_recommendations)

    # Retrieve and format wine recommendations
    recommended_wine_ids = features_df.iloc[indices[0]]['WineID'].tolist()
    recommended_wines = features_df[features_df['WineID'].isin(recommended_wine_ids)][
        ['WineID', 'WineName', 'Country', 'Type', 'RegionName', 'Body', 'Acidity', 'Grapes_list', 'avg_rating']
    ]

    # Add similarity scores (1 - distance)
    distance_map = {wine_id: dist for wine_id, dist in zip(recommended_wine_ids, distances[0])}
    recommended_wines['Similarity'] = recommended_wines['WineID'].map(lambda wine_id: 1 - distance_map[wine_id])

    # Sort by similarity and return
    return recommended_wines.sort_values('Similarity', ascending=False)
