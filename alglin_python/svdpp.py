from math import sqrt
import numpy as np



def train_svdpp(df, train, n_factors, lr=0.05, reg=0.02, miter=10):
    global_mean = train['rating'].mean()
    n_users = df['userId'].max() + 1
    n_items = df['movieId'].max() + 1
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    p = np.random.normal(0.1, 0.1, (n_users, n_factors))
    q = np.random.normal(0.1, 0.1, (n_items, n_factors))
    implicit_factors = np.random.normal(0.1, 0.1, (n_items, n_factors))

    error = []

    for t in range(miter):
        sq_error = 0

        implicit_sum = np.zeros((n_users, n_factors))
        for row in train.itertuples():
            # Access tuple attributes using dot notation
            u = row.userId
            i = row.movieId
            r_ui = row.rating

            temp_p = p[u]
            temp_q = q[i]

            implicit_items = train[train['userId'] == u]['movieId'].values
            implicit_sum[u] = np.sum(implicit_factors[implicit_items], axis=0)
            pred = global_mean + bu[u] + bi[i] + np.dot(temp_p + implicit_sum[u], temp_q)
            e_ui = r_ui - pred
            sq_error += e_ui ** 2

            # Update biases
            bu[u] += lr * (e_ui - reg * bu[u])
            bi[i] += lr * (e_ui - reg * bi[i])

            # Update factors
            p[u] += lr * (e_ui * temp_q - reg * temp_p)
            q[i] += lr * (e_ui * temp_p - reg * temp_q)

            # Update implicit factors
            for j in implicit_items:
                implicit_factors[j] += lr * (e_ui * q[i] - reg * implicit_factors[j])

        error.append(sqrt(sq_error / len(train)))

    return global_mean, bu, bi, p, q, implicit_factors, error


def predict_svdpp(train, global_mean, bu, bi, p, q, implicit_factors, user_id, movie_id):
    if user_id >= len(bu) or movie_id >= len(bi):
        return global_mean

    implicit_items = train[train['userId'] == user_id]['movieId'].values
    implicit_sum = np.sum(implicit_factors[implicit_items], axis=0) if len(implicit_items) > 0 else np.zeros(p.shape[1])

    prediction = global_mean + bu[user_id] + bi[movie_id] + np.dot(p[user_id] + implicit_sum, q[movie_id])
    return prediction


def evaluate_model_svdpp(train, global_mean, bu, bi, p, q, implicit_factors, test_data):
    predictions = []
    actuals = []

    for row in test_data.itertuples():
        user_id = row.userId
        movie_id = row.movieId
        actual_rating = row.rating

        predicted_rating = predict_svdpp(train, global_mean, bu, bi, p, q, implicit_factors, user_id, movie_id)

        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f'RMSE: {rmse}')

    return predictions, actuals
