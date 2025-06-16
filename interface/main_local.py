import pickle
import os
import pandas as pd
from pathlib import Path
import ipdb

from cv_functions.recommendation import get_wine_recommendations_by_characteristics
from cv_functions.data import  get_data_with_cache
from cv_functions.model import train_model, load_model
from cv_functions.encoder import Encoder_features_fit_transform, Encoder_features_transform
# from transformers.ratings_stat import RatingsStatsAggregator
from transformers.ratings_agg import Rates_aggregator


# define paths
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
clean_wine_path = os.path.join(LOCAL_DATA_PATH, "clean_wine_knn.csv") # search for file for knn if existed
LOCAL_MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
preprocessor_path = os.path.join(LOCAL_MODEL_PATH, "preprocessor.pkl")
Pipeline_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "models")

# only process the cleaning and preprocessing if the final feature dataFrame is not available)
if Path(clean_wine_path).is_file():
     wine_scaled_df = pd.read_csv(clean_wine_path)

else:
    # load the cleaned data (clean data and save csv if not exist)
    wines_clean_df, ratings_clean_df = get_data_with_cache(LOCAL_DATA_PATH)

    ratings_stats = Rates_aggregator(ratings_clean_df)

    # merge DataFrames
    wine_df = wines_clean_df.merge(ratings_stats, on = 'WineID')  # ratings_stats already contains the merged DataFrame

    print('save merged file to .csv as lookup table....')
    save_path =  os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    wine_df.to_csv(os.path.join(save_path, 'wine_lookup.csv'), index=False)

    # keep columns for KNN models
    drop_columns = ['WineID', 'WineName','Elaborate','Grapes', 'Harmonize', 'Code', 'Country','RegionID', 'RegionName',
                    'WineryID','WineryName','Website','Vintages']
    wine_scaled_df_knn = wine_df.drop(columns=drop_columns, axis=1)

    # ipdb.set_trace()
    if Path(preprocessor_path).is_file(): # if there's already a preprocessor file, load it
        wine_scaled_df = Encoder_features_transform(wine_scaled_df_knn)
    else:
        wine_scaled_df = Encoder_features_fit_transform(wine_scaled_df_knn)

    print('save cleaned merged file for knn to .csv ....')
    save_path =  os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    wine_scaled_df.to_csv(os.path.join(save_path, 'clean_wine_knn.csv'), index=False)

# step4 : load model ( or train model if not existed)
#ipdb.set_trace()
LOCAL_MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
pickle_file_path = os.path.join(LOCAL_MODEL_PATH, "trained_model.pkl")

if not Path(pickle_file_path).is_file():
    print("Model file not found.")
    model = train_model(wine_scaled_df)
