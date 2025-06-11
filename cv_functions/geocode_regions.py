
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def geocode_regions(df, region_column='RegionName',country_column = 'Country', cache_file='geocoding_cache.pkl', min_delay=1.0):
    """
    Geocode region names with caching and rate limiting

    Parameters:
        df (pd.DataFrame): Input DataFrame
        region_column (str): Name of region column
        cache_file (str): Path to cache file
        min_delay (float): Minimum delay between API requests (seconds)

    Returns:
        pd.DataFrame: DataFrame with latitude/longitude columns
    """
    # Initialize geocoder with rate limiter
    geolocator = Nominatim(user_agent="regional_analysis_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=min_delay)

    # create coutry vs city dictionary
    df_dict_df = df[[region_column,country_column]].drop_duplicates()
    df_dict = pd.Series(df_dict_df[country_column].values,index=df_dict_df[region_column]).to_dict()

    # Load existing cache
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            geocode_cache = pickle.load(f)
    else:
        geocode_cache = {}

    # Get unique regions needing geocoding
    unique_regions = df[region_column].dropna().unique()

    new_regions = [r for r in unique_regions if r not in geocode_cache]

    print(f"Found {len(unique_regions)} unique regions ({len(new_regions)} new)")

    # Geocode new regions with retry logic
    for region in new_regions:
        for attempt in range(3):  # 3 retry attempts
            try:
                location = geocode(region)
                geocode_cache[region] = {
                    'latitude': location.latitude if location else np.nan,
                    'longitude': location.longitude if location else np.nan
                }

                if location.latitude == 'none':
                    location = geocode(df_dict[region])
                    geocode_cache[region] = {
                    'latitude': location.latitude if location else np.nan,
                    'longitude': location.longitude if location else np.nan}

                break
            except Exception as e:
                if attempt == 2:  # Final attempt failed
                    geocode_cache[region] = {'latitude': np.nan, 'longitude': np.nan}
                    print(f"Failed to geocode {region} after 3 attempts")
                continue


    # Save updated cache
    with open(cache_path, 'wb') as f:
        pickle.dump(geocode_cache, f)

    # Create coordinate columns
    df['latitude'] = df[region_column].map(lambda x: geocode_cache.get(x, {}).get('latitude'))
    df['longitude'] = df[region_column].map(lambda x: geocode_cache.get(x, {}).get('longitude'))

    return df #.drop(columns=[region_column])
