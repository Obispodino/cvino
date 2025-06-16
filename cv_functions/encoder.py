import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from transformers.top_k_encoder import TopNGrapeOneHotEncoder
from transformers.body_ordinal_encoder import BodyOrdinalEncoder
from transformers.acid_ordinal_encoder import AcidOrdinalEncoder
from transformers.ratings_stat import RatingsStatsAggregator
import ipdb


LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
preprocessor_file = os.path.join(LOCAL_DATA_PATH, "preprocessor.pkl")


# Function to get column names
def get_feature_names_out(ct):
    names = []
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, 'get_feature_names_out'):
            names.extend(trans.get_feature_names_out())
        else:
            names.extend(cols)
    return names

def Encoder_features_fit_transform(df:pd.DataFrame):
    '''
    encode features

    '''
    body_categories = [['Very light-bodied', 'Light-bodied', 'Medium-bodied', 'Full-bodied', 'Very full-bodied']]
    acidity_categories = [['Low', 'Medium', 'High']]
    numeric_features = ['ABV','latitude', 'longitude', 'avg_rating', 'rating_count', 'rating_std']

    type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    body_encoder_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Medium-bodied')),
    ('ordinal', OrdinalEncoder(categories=body_categories))
    ])

    acidity_encoder_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Medium')),
    ('ordinal', OrdinalEncoder(categories=acidity_categories))
    ])

    # make sure number columns have medium for missing values
    numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # or 'mean'
    ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('Type', type_encoder, ['Type']),
        ('Grape', TopNGrapeOneHotEncoder(top_n=60), ['Grapes_list']),
        ('Body', body_encoder_pipeline, ['Body']),
        ('Acidity', acidity_encoder_pipeline, ['Acidity']),
        ('num', numeric_pipeline, numeric_features),
    ],
    remainder='passthrough',
    n_jobs=-1,
    )

    #preprocessor.set_output(transform='pandas')

    preprocessor.fit(df)

    #save preprocessor into pickle
    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessor, f)


    df_processed = preprocessor.transform(df)
    print("Shape of transformed array:", df_processed.shape)

    #columns_names = preprocessor.get_feature_names_out()

    columns_names = get_feature_names_out(preprocessor)
    print("Number of column names:", len(columns_names))
    print("Difference in columns:", df_processed.shape[1] - len(columns_names))

    # change column names
    columns_names = ['Body_encoded' if col == 'Body' else col for col in columns_names]
    columns_names = ['Acidity_encoded' if col == 'Acidity' else col for col in columns_names]
    X_df = pd.DataFrame(df_processed, columns=columns_names, index=df.index)

    return X_df

def Encoder_features_transform(df:pd.DataFrame):

    with open(preprocessor_file, 'rb') as f:
        preprocessor = pickle.load(f)

    #preprocessor.set_output(transform='pandas')
    df_processed = preprocessor.transform(df)
    column_names = get_feature_names_out(preprocessor)
    # change column names
    column_names = ['Body_encoded' if col == 'Body' else col for col in column_names]
    column_names = ['Acidity_encoded' if col == 'Acidity' else col for col in column_names]
    X_df = pd.DataFrame(df_processed, columns=column_names, index=df.index)

    return X_df
