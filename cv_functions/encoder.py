import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from transformers.top_k_encoder import TopKOneHotEncoder
from transformers.body_ordinal_encoder import BodyOrdinalEncoder
from transformers.acid_ordinal_encoder import AcidOrdinalEncoder

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
preprocessor_file = os.path.join(LOCAL_DATA_PATH, "preprocessor.pkl")


def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if transformer == 'drop':
            continue
        elif transformer == 'passthrough':
            feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            if isinstance(columns, list):
                names = transformer.get_feature_names_out(columns)
            else:
                names = transformer.get_feature_names_out([columns])
            feature_names.extend(names)
        elif hasattr(transformer, 'top_grapes_'):
            # Custom: TopKGrapeEncoder
            safe_grapes = [grape.replace(" ", "_") for grape in transformer.top_grapes_]
            names = [f'Grape_{g}' for g in safe_grapes]
            feature_names.extend(names)
        elif hasattr(transformer, 'categories'):
            feature_names.extend(columns)
        else:
            if isinstance(columns, list):
                feature_names.extend(columns)
            else:
                feature_names.append(columns)

    return feature_names

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
    numeric_features = ['ABV','latitude', 'longitude']

    # make sure number columns have medium for missing values
    numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # or 'mean'
    ('scaler', MinMaxScaler())
    ])

    type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
    transformers=[
        ('Type', type_encoder, ['Type']),
        ('Grape', TopKOneHotEncoder(top_k=10), ['Grapes_list']),
        ('Body', BodyOrdinalEncoder(column='Body', categories=body_categories, fallback='Medium-bodied',output_name='Body_encoded'), ['Body']),
        ('Acidity', AcidOrdinalEncoder(column='Acidity', categories=acidity_categories, fallback='Medium',output_name='Acidity_encoded'), ['Acidity']),
        ('num', numeric_pipeline, numeric_features)
    ],
    remainder='passthrough',
    n_jobs=-1,
    )

    preprocessor.fit(df)
    #save preprocessor into pickle

    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessor, f)

    df_processed = preprocessor.transform(df)

    columns_names = get_feature_names_out(preprocessor)

    # change column names
    column_names = ['Body_encoded' if col == 'Body' else col for col in column_names]
    column_names = ['Acidity_encoded' if col == 'Acidity' else col for col in column_names]
    X_df = pd.DataFrame(df_processed, columns=column_names, index=df.index)

    return X_df

def Encoder_features_transform(df:pd.DataFrame):

    with open(preprocessor_file, 'rb') as f:
        preprocessor = pickle.load(f)
    df_processed = preprocessor.transform(df)
    column_names = get_feature_names(preprocessor)
    # change column names
    column_names = ['Body_encoded' if col == 'Body' else col for col in column_names]
    column_names = ['Acidity_encoded' if col == 'Acidity' else col for col in column_names]
    X_df = pd.DataFrame(df_processed, columns=column_names, index=df.index)

    return X_df
