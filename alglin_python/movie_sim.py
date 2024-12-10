import numpy as np
from math import sqrt

def get_movie_user_ids(df, movieId):
    """
    Get the list of user IDs who rated the given movie.
    """
    if movieId not in df['movieId'].values:
        return []
    return df.loc[df['movieId'] == movieId, 'userId'].tolist()

def get_movie_ratings(df, movieId):
    """
    Get the list of ratings for the given movie.
    """
    if movieId not in df['movieId'].values:
        return []
    return df.loc[df['movieId'] == movieId, 'rating'].tolist()

def get_rating(df, movieId, userId):
    """
    Get the rating given by a user to a specific movie.
    """
    rating = df.loc[(df['movieId'] == movieId) & (df['userId'] == userId), 'rating']
    return rating.iloc[0] if not rating.empty else 0

def pearson_similarity(df, movieId1, movieId2):
    """
    Compute Pearson similarity between two movies based on ratings.
    """
    users1 = get_movie_user_ids(df, movieId1)
    users2 = get_movie_user_ids(df, movieId2)
    common_users = list(set(users1) & set(users2))

    if len(common_users) == 0:
        return 0

    ratings1 = [get_rating(df, movieId1, user) for user in common_users]
    ratings2 = [get_rating(df, movieId2, user) for user in common_users]

    mean1 = np.mean(ratings1)
    mean2 = np.mean(ratings2)

    numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))
    denominator = sqrt(sum((r1 - mean1) ** 2 for r1 in ratings1)) * sqrt(sum((r2 - mean2) ** 2 for r2 in ratings2))

    return numerator / denominator if denominator != 0 else 0

def cosine_similarity(df, movieId1, movieId2):
    """
    Compute Cosine similarity between two movies based on ratings.
    """
    users1 = get_movie_user_ids(df, movieId1)
    users2 = get_movie_user_ids(df, movieId2)
    common_users = list(set(users1) & set(users2))

    if len(common_users) == 0:
        return 0

    ratings1 = [get_rating(df, movieId1, user) for user in common_users]
    ratings2 = [get_rating(df, movieId2, user) for user in common_users]

    numerator = sum(r1 * r2 for r1, r2 in zip(ratings1, ratings2))
    denominator = sqrt(sum(r1 ** 2 for r1 in ratings1)) * sqrt(sum(r2 ** 2 for r2 in ratings2))

    return numerator / denominator if denominator != 0 else 0
