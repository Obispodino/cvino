import pandas as pd
import numpy as np
import os
from pathlib import Path
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder

def wine_clean_features(wd: pd.DataFrame) -> pd.DataFrame:
    '''
    this function clean the input wine dataset and rating dataset
    input: wine dataset in pd.DataFrame format, ratings dataset in pd.DataFrame format
    '''

    # DATA CLEANING STAGE : Clean Wine Dataset
    print("=== CLEANING WINES DATASET ===")

    # Create a clean copy
    wines_clean = wd.copy()

    # 1. Handle missing values
    # Drop rows with missing essential information
    wines_clean = wines_clean.dropna(subset=['WineName', 'Country', 'Type'])

    # Fill missing ABV with median by wine type
    wines_clean['ABV'] = wines_clean.groupby('Type')['ABV'].transform(lambda x: x.fillna(x.median()))

    # Fill missing RegionName with 'Unknown'
    # wines_clean['RegionName'] = wines_clean['RegionName'].fillna('Unknown')

    # Fill missing WineryName with 'Unknown'
    wines_clean['WineryName'] = wines_clean['WineryName'].fillna('Unknown')

    # 2. Standardize text fields
    print("Standardizing text fields...")
    wines_clean['Country'] = wines_clean['Country'].str.strip()
    wines_clean['Type'] = wines_clean['Type'].str.strip()
    wines_clean['WineName'] = wines_clean['WineName'].str.strip()

    print("Wine dataset cleaning completed!")

    return wines_clean

def ratings_clean_features(ratings: pd.DataFrame, wines_clean: pd.DataFrame) -> pd.DataFrame:

    # DATA CLEANING STAGE : Clean Ratings Dataset
    print("=== CLEANING RATINGS DATASET ===")

    # Create a clean copy
    ratings_clean = ratings.copy()

    # 1. Remove invalid ratings
    print("Removing invalid ratings...")

    # Remove ratings outside valid range (assuming 1-5 scale)
    ratings_clean = ratings_clean[(ratings_clean['Rating'] >= 1) & (ratings_clean['Rating'] <= 5)]

    # Remove entries with missing UserID or WineID
    ratings_clean = ratings_clean.dropna(subset=['UserID', 'WineID'])

    # 2. Filter ratings for wines that exist in our clean wine dataset
    valid_wine_ids = set(wines_clean['WineID'])
    ratings_clean = ratings_clean[ratings_clean['WineID'].isin(valid_wine_ids)]

    print("Ratings dataset cleaning completed!")

    return  ratings_clean


def preprocess_features(features_df: pd.DataFrame, ratings_clean: pd.DataFrame, N_TOP_GRAPES) -> pd.DataFrame:
    '''
    this function preprocess the features from wine dataset including scaling,
    encode featuers and merge rating dataframe to wine dataframe
    additional features including 'avg_rating', 'rating_count', 'rating_std'
    '''
    # DATA CLEANING STAGE - Step 4: Prepare Features for k-NN Model
    print("=== PREPARING FEATURES FOR k-NN MODEL ===")

    # 1. Process categorical variables
    print("Processing categorical variables...")

    # a. OneHotEncode Type column
    type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    type_encoded = type_encoder.fit_transform(features_df[['Type']])
    type_columns = [f'Type_{cat}' for cat in type_encoder.categories_[0]]
    type_df = pd.DataFrame(type_encoded, columns=type_columns, index=features_df.index)

    # b. Process Grapes (extract top N grapes)
    N_TOP_GRAPES = N_TOP_GRAPES  # Consider top 50 grape varieties

    # Get most common grape varieties
    all_grapes = [grape for sublist in features_df['Grapes_list'] for grape in sublist if isinstance(sublist, list)]
    top_grapes = pd.Series(all_grapes).value_counts().head(N_TOP_GRAPES).index.tolist()

    # Create binary columns for each top grape
    for grape in top_grapes:
        features_df[f'Grape_{grape}'] = features_df['Grapes_list'].apply(
            lambda x: 1 if isinstance(x, list) and grape in x else 0
        )

    grape_columns = [f'Grape_{grape}' for grape in top_grapes]

    # c. Ordinal encode Body and Acidity
    # Define the order for Body
    body_categories = ['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']
    body_encoder = OrdinalEncoder(categories=[body_categories])
    features_df['Body_encoded'] = np.nan
    body_mask = features_df['Body'].isin(body_categories)
    if body_mask.any():
        features_df.loc[body_mask, 'Body_encoded'] = body_encoder.fit_transform(
            features_df.loc[body_mask, ['Body']])
    features_df['Body_encoded'] = features_df['Body_encoded'].fillna(2)  # Default to medium-bodied

    # Define the order for Acidity
    acidity_categories = ['Low', 'Medium', 'High']
    acidity_encoder = OrdinalEncoder(categories=[acidity_categories])
    features_df['Acidity_encoded'] = np.nan
    acidity_mask = features_df['Acidity'].isin(acidity_categories)
    if acidity_mask.any():
        features_df.loc[acidity_mask, 'Acidity_encoded'] = acidity_encoder.fit_transform(
            features_df.loc[acidity_mask, ['Acidity']])
    features_df['Acidity_encoded'] = features_df['Acidity_encoded'].fillna(1)  # Default to medium acidity

    # 2. Handle numerical features
    print("Processing numerical features...")
    # Ensure ABV is numeric
    features_df['ABV_numeric'] = pd.to_numeric(features_df['ABV'], errors='coerce')
    features_df['ABV_numeric'] = features_df['ABV_numeric'].fillna(features_df['ABV_numeric'].median())

    # Keep latitude and longitude as is (they're already numeric)
    numeric_columns = ['ABV_numeric', 'latitude', 'longitude']

    # 3. Create wine profile features from aggregated ratings
    print("Creating wine profile features...")
    wine_stats = ratings_clean.groupby('WineID').agg({
        'Rating': ['mean', 'count', 'std']
    }).round(2)

    wine_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
    wine_stats = wine_stats.fillna(0)


    # Merge with wine features
    features_df = features_df.merge(wine_stats, left_on='WineID', right_index=True, how='left')
    features_df[['avg_rating', 'rating_count', 'rating_std']] = features_df[['avg_rating', 'rating_count', 'rating_std']].fillna(0)
    rating_columns = ['avg_rating', 'rating_count', 'rating_std']

    # 4. Select final feature columns for k-NN
    print("Compiling final feature set...")
    # Combine all feature columns
    feature_columns = numeric_columns + ['Body_encoded', 'Acidity_encoded'] + rating_columns + grape_columns
    feature_dfs = [features_df[feature_columns], type_df]

    # Create final feature matrix
    X = pd.concat(feature_dfs, axis=1)

    # 5. Scale features
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=features_df.index)

    print(f"Feature matrix shape: {X_scaled_df.shape}")
    print(f"Feature columns: {len(X_scaled_df.columns)} columns")
    print("\nFeature matrix ready for k-NN model!")

    # Update the feature_columns variable to include all columns
    feature_columns = X_scaled_df.columns.tolist()

    return X_scaled_df
