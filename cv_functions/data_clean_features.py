import pandas as pd


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
