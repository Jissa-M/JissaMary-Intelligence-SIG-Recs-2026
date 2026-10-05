import pandas as pd

ratings=pd.read_csv("ratings.csv")
print(ratings.head())
print(ratings.shape)
print(ratings.columns)
print(ratings.info())
print(ratings.isnull().sum())

print("Number of users:", ratings["userId"].nunique())
print("Number of movies:", ratings["movieId"].nunique())
print("Number of ratings:", len(ratings))

print(ratings["rating"].value_counts().sort_index())

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
print(train_ratings.head())
print(test_ratings.head())

#BUILD THE MATRIX
train_matrix = train_ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

print(train_matrix.shape)
print(train_matrix.head())
##
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(
    train_matrix.fillna(0)
)

print(similarity_matrix.shape)

print(similarity_matrix[:5, :5])

def predict_rating(user_id, movie_id, k=5):
    if movie_id not in train_matrix.columns:
        return None

    user_index = train_matrix.index.get_loc(user_id)

    similarities = similarity_matrix[user_index]

    similar_users = pd.Series(
        similarities,
        index=train_matrix.index
    )

    similar_users = similar_users.drop(user_id)

    similar_users = similar_users.sort_values(
        ascending=False
    )

    top_users = similar_users.head(k)

    ratings = train_matrix.loc[
        top_users.index,
        movie_id
    ].dropna()

    top_users = top_users.loc[ratings.index]

    if len(ratings) == 0:
        return None

    prediction = (
        (top_users * ratings).sum()/ top_users.sum()
    )

    return prediction
print(predict_rating(1, 50))
from sklearn.metrics import mean_squared_error
import numpy as np

actual = []
predicted = []

for _, row in test_ratings.iterrows():

    prediction = predict_rating(
        row["userId"],
        row["movieId"]
    )

    if prediction is not None:
        actual.append(row["rating"])
        predicted.append(prediction)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

print("Memory based CF RMSE:", rmse)
