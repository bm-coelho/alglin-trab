import numpy as np
import pandas as pd


def svd_recommendation(df, n_factors=20):
    """
    Performs Singular Value Decomposition (SVD) for a recommendation system.

    Parameters:
    - df: DataFrame with columns ['userId', 'movieId', 'rating'].
    - n_factors: Number of latent factors to retain (default is 20).

    Returns:
    - user_factors: Matrix of user latent factors.
    - item_factors: Matrix of item latent factors.
    - global_mean: Global mean of the ratings.
    - user_mapping: Dictionary mapping user IDs to indices.
    - movie_mapping: Dictionary mapping movie IDs to indices.
    """
    # Get the unique users and items
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    # Map users and movies to indices
    user_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_mapping = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    # Create the ratings matrix
    n_users = len(user_ids)
    n_items = len(movie_ids)
    ratings_matrix = np.zeros((n_users, n_items))

    for row in df.itertuples():
        u = user_mapping[row.userId]
        i = movie_mapping[row.movieId]
        ratings_matrix[u, i] = row.rating

    # Center the ratings by subtracting the global mean
    global_mean = ratings_matrix[ratings_matrix > 0].mean()
    centered_matrix = ratings_matrix - global_mean
    centered_matrix[ratings_matrix == 0] = 0  # Keep missing values as 0

    # Perform Singular Value Decomposition
    U, sigma, VT = np.linalg.svd(centered_matrix, full_matrices=False)

    # Retain only the top n_factors
    sigma = np.diag(sigma[:n_factors])
    U = U[:, :n_factors]
    VT = VT[:n_factors, :]

    # Compute the user and item factors
    user_factors = np.dot(U, sigma)
    item_factors = VT.T

    return user_factors, item_factors, global_mean, user_mapping, movie_mapping


def predict_svd(user_factors, item_factors, global_mean, user_id, movie_id, user_mapping, movie_mapping):
    """
    Predicts the rating for a given user and movie using SVD factors.

    Parameters:
    - user_factors: User latent factors matrix.
    - item_factors: Item latent factors matrix.
    - global_mean: Global mean of the ratings.
    - user_id: ID of the user.
    - movie_id: ID of the movie.
    - user_mapping: Dictionary mapping user IDs to indices.
    - movie_mapping: Dictionary mapping movie IDs to indices.

    Returns:
    - Predicted rating.
    """
    if user_id not in user_mapping or movie_id not in movie_mapping:
        return global_mean

    u = user_mapping[user_id]
    i = movie_mapping[movie_id]

    return global_mean + np.dot(user_factors[u], item_factors[i])


def evaluate_model_svd(df, test_data, user_factors, item_factors, global_mean, user_mapping, movie_mapping):
    """
    Evaluates the SVD model using RMSE on test data.

    Parameters:
    - df: DataFrame with training data.
    - test_data: DataFrame with columns ['userId', 'movieId', 'rating'].
    - user_factors: User latent factors matrix.
    - item_factors: Item latent factors matrix.
    - global_mean: Global mean of the ratings.
    - user_mapping: Dictionary mapping user IDs to indices.
    - movie_mapping: Dictionary mapping movie IDs to indices.

    Returns:
    - rmse: Root Mean Square Error on the test set.
    """
    predictions = []
    actuals = []

    for row in test_data.itertuples():
        user_id = row.userId
        movie_id = row.movieId
        actual_rating = row.rating

        predicted_rating = predict_svd(user_factors, item_factors, global_mean, user_id, movie_id, user_mapping, movie_mapping)

        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f"RMSE: {rmse}")

    return rmse
