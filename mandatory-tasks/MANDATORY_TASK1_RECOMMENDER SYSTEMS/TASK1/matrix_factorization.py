import pandas as pd
import numpy as np

ratings = pd.read_csv("ratings.csv")

print(ratings.head())
print(ratings.shape)

from sklearn.model_selection import train_test_split

train_list = []
test_list = []

for user_id, user_ratings in ratings.groupby("userId"):

    train, test = train_test_split(
        user_ratings,
        test_size=0.2,
        random_state=42
    )

    train_list.append(train)
    test_list.append(test)

train_ratings = pd.concat(train_list)
test_ratings = pd.concat(test_list)

print("Training ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))

user_ids = train_ratings["userId"].unique()
movie_ids = train_ratings["movieId"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

print("Number of users:", len(user_to_index))
print("Number of movies:", len(movie_to_index))

k = 20

num_users = len(user_to_index)
num_movies = len(movie_to_index)

U = np.random.normal(0,0.1,
    (num_users, k)
)

V = np.random.normal(
    0,
    0.1,
    (num_movies, k)
)

print("U shape:", U.shape)
print("V shape:", V.shape)

user_id = 1
movie_id = 50

user_index = user_to_index[user_id]
movie_index = movie_to_index[movie_id]

user_vector = U[user_index]
movie_vector = V[movie_index]

prediction = np.dot(
    user_vector,
    movie_vector
)

print("Predicted rating:", prediction)

learning_rate = 0.01
epochs = 10

for epoch in range(epochs):

    total_error = 0

    for _, row in train_ratings.iterrows():

        user_id = row["userId"]
        movie_id = row["movieId"]
        actual_rating = row["rating"]

        user_index = user_to_index[user_id]
        movie_index = movie_to_index[movie_id]

        prediction = np.dot(
            U[user_index],
            V[movie_index]
        )

        error = actual_rating - prediction

        user_vector = U[user_index].copy()
        movie_vector = V[movie_index].copy()

        U[user_index] = (
            user_vector
            + learning_rate * error * movie_vector
        )

        V[movie_index] = (
            movie_vector
            + learning_rate * error * user_vector
        )

        total_error += error ** 2

    mse = total_error / len(train_ratings)

    print("Epoch:", epoch + 1, "MSE:", mse)
    
    from sklearn.metrics import mean_squared_error

actual = []
predicted = []

for _, row in test_ratings.iterrows():

    user_id = row["userId"]
    movie_id = row["movieId"]

    if user_id not in user_to_index:
        continue

    if movie_id not in movie_to_index:
        continue

    user_index = user_to_index[user_id]
    movie_index = movie_to_index[movie_id]

    prediction = np.dot(
        U[user_index],
        V[movie_index]
    )

    actual.append(row["rating"])
    predicted.append(prediction)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

print("Matrix Factorization RMSE:", rmse)




#comparison
print("\nFinal Comparison")
print("---------------------------")
print("Memory-based CF RMSE:", 1.090562677919281)
print("Matrix Factorization RMSE:", rmse)

#calculate the improvement
memory_rmse = 1.090562677919281
mf_rmse = rmse

improvement = (
    (memory_rmse - mf_rmse)
    / memory_rmse
) * 100

print("RMSE improvement:", improvement, "%")

#plots
import matplotlib.pyplot as plt

methods = [
    "Memory-based CF",
    "Matrix Factorization"
]

rmse_values = [
    memory_rmse,
    mf_rmse
]

plt.bar(methods, rmse_values)

plt.ylabel("RMSE")
plt.title("Recommender Model Comparison")

plt.show()