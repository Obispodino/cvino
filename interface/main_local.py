import pickle
import os
import pandas as pd
from pathlib import Path
import ipdb

from cv_functions.recommendation import get_wine_recommendations_by_characteristics
from cv_functions.data import  get_data_with_cache
from cv_functions.model import train_model, load_model
from cv_functions.encoder import Encoder_features_fit_transform, Encoder_features_transform


# define paths
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
clean_wine_path = os.path.join(LOCAL_DATA_PATH, "cleaned_wine.csv") # search for clean file if existed
LOCAL_MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
preprocessor_path = os.path.join(LOCAL_MODEL_PATH, "preprocessor.pkl")

# only process the cleaning and preprocessing if the final feature dataFrame is not available)
if Path(clean_wine_path).is_file():
    wine_scaled_df = pd.read_csv(clean_wine_path)

else:
    # load the cleaned data (clean data and save csv if not exist)
    wines_clean_df, ratings_clean_df = get_data_with_cache(LOCAL_DATA_PATH)
    # preprocess_features for wine_df with pipeline
    Pipeline_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "models")

    if Path(preprocessor_path).is_file():
        wine_proccessed = Encoder_features_transform(wines_clean_df)
    else:
        wine_proccessed = Encoder_features_fit_transform(wines_clean_df)
    # preprocess rating features
    ratings_stats = ratings_clean_df.groupby('WineID').agg({
    'Rating': ['mean', 'count', 'std']}).round(2)

    ratings_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
    ratings_stats = ratings_stats.fillna(0)
    ipdb.set_trace()
    # merge DataFrames
    wine_scaled_df = wine_proccessed.merge(ratings_stats, left_on='WineID', right_index=True, how='left')
    print('save cleaned file to .csv ....')

    save_path =  os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    wine_scaled_df.to_csv(os.path.join(save_path, 'wines_clean_preprocessed.csv'), index=False)


# step4 : load model ( or train model if not existed)
# load model
LOCAL_MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
pickle_file_path = os.path.join(LOCAL_MODEL_PATH, "trained_model.pkl")

if Path(pickle_file_path).is_file():
    print("Model file exists.")
    model = load_model()
else:
    print("Model file not found.")
    model = train_model(wine_scaled_df)

# def predict():
#     pass
