from sklearn.neighbors import NearestNeighbors
import pickle
import os

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino","models")
pickle_file = os.path.join(LOCAL_DATA_PATH, "trained_model.pkl")

def train_model(X_scaled_df, n_neighbors = 6):
    """
    BUILDING k-NN RECOMMENDATION MODEL
    """
    print("=== BUILDING k-NN RECOMMENDATION MODEL ===")
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn_model.fit(X_scaled_df)

    with open(pickle_file, 'wb') as f:
        pickle.dump(knn_model, f)

    return knn_model


def load_model(filepath=pickle_file):
    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)
    return model
