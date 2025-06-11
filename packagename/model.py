from sklearn.neighbors import NearestNeighbors



def train_model(X_scaled_df, n_neighbors = 6):
    """
    BUILDING k-NN RECOMMENDATION MODEL
    """
    print("=== BUILDING k-NN RECOMMENDATION MODEL ===")
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn_model.fit(X_scaled_df)

    return knn_model
