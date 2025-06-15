import os
import pandas as pd

from pathlib import Path


from cv_functions.geocode_regions import geocode_regions
from cv_functions.data_clean_features import wine_clean_features, ratings_clean_features

N_TOP_GRAPES = 50

def get_data_with_cache(cache_path:Path) -> pd.DataFrame:
    # LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    clean_wine_path = os.path.join(cache_path, "wines_clean.csv")
    clean_ratings_path = os.path.join(cache_path, "ratings_clean.csv")

    if Path(clean_wine_path).is_file() and Path(clean_ratings_path).is_file():
        print( "\nLoad clean wine data from local CSV..." )
        wines_clean_df = pd.read_csv(clean_wine_path)
        wines_clean_df['Grapes_list'] =  wines_clean_df['Grapes'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        print( "\nLoad clean rating data from local CSV..." )
        ratings_clean_df = pd.read_csv(clean_ratings_path)

    else:
        print("\nLoad raw data and preprocess...")
        # load data
        Raw_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data", "last")
        wines_file = os.path.join(Raw_DATA_PATH, "XWines_Full_100K_wines.csv")
        wines_data = pd.read_csv(wines_file)

        ratings_file = os.path.join(Raw_DATA_PATH, "XWines_Full_21M_ratings.csv")
        ratings_data = pd.read_csv(ratings_file)
        # step1 : geocode the region
        wd = geocode_regions(wines_data, min_delay=2.0)

        # step2 : clean_features (from preprocessor)
        wines_clean_df = wine_clean_features(wd)
        ratings_clean_df = ratings_clean_features(ratings_data, wines_clean_df)

        save_path =  os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
        wines_clean_df.to_csv(os.path.join(save_path, 'wines_clean.csv'), index=False)
        ratings_clean_df.to_csv(os.path.join(save_path, 'ratings_clean.csv'), index=False)
        wines_clean_df['Grapes_list'] =  wines_clean_df['Grapes'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    return wines_clean_df, ratings_clean_df
