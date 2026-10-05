# Task 01 — Collaborative Filtering

## 1. Objective

The objective of this task is to implement and compare two collaborative
filtering approaches:

1. Memory-Based Collaborative Filtering
2. Model-Based Collaborative Filtering using Matrix Factorization

Both approaches are evaluated using the same dataset and the same
train/test split.

The goal is to understand the difference between directly using user
similarities and learning latent representations of users and items.

---

## 2. Dataset

### Dataset Used

**MovieLens Latest-Small**

The MovieLens dataset was selected because it naturally represents a
user-item interaction problem.

- Users represent users of the recommendation system.
- Movies represent items.
- Ratings represent interactions between users and movies.

This makes the dataset suitable for implementing and evaluating
collaborative filtering algorithms.

### Dataset Statistics

| Property | Value |
|---|---:|
| Number of users | 610 |
| Number of movies | 9,724 |
| Number of ratings | 100,836 |
| Rating range | 0.5 – 5.0 |

The dataset contains the following columns:

```text
userId
movieId
rating
timestamp
```

No missing values were found in the dataset.

---

## 3. Dataset Sparsity

The complete user-item matrix contains:

$$
610 \times 9724 = 5,931,640
$$

possible user-movie interactions.

Only 100,836 interactions are observed.

Therefore:

- Observed interactions ≈ **1.7%**
- Unobserved interactions ≈ **98.3%**

This high sparsity is representative of the type of problem collaborative
filtering methods are designed to address.

---

# 4. Data Preprocessing

The ratings were divided into training and testing data using a
**per-user 80/20 split**.

For each user:

- 80% of their ratings were used for training.
- 20% of their ratings were used for testing.

A fixed random seed of `42` was used for reproducibility.

### Train/Test Split

| Dataset | Number of Ratings |
|---|---:|
| Training | 80,419 |
| Testing | 20,417 |

The same train/test split was used for both approaches to ensure a fair
comparison.

---

# 5. Memory-Based Collaborative Filtering

## 5.1 Approach

The first approach uses **user-user collaborative filtering**.

The training data was converted into a user-item rating matrix.

```text
Rows    → Users
Columns → Movies
Values  → Ratings
```

The resulting training matrix contained:

```text
610 users × 8965 movies
```

Only movies that appeared in the training data were included in this
matrix.

---

## 5.2 User Similarity

**Cosine similarity** was used to calculate the similarity between users.

$$
similarity(A,B)=
\frac{A\cdot B}
{\|A\|\|B\|}
$$

A higher similarity indicates that two users have more similar rating
patterns.

The resulting similarity matrix had dimensions:

```text
610 × 610
```

Missing ratings in the user-item matrix were filled with zero before
calculating cosine similarity.

---

## 5.3 Rating Prediction

For a given user and movie, the prediction process was:

1. Find the target user's similarity with all other users.
2. Remove the target user.
3. Select the top 5 most similar users.
4. Keep only similar users who have rated the target movie.
5. Calculate a similarity-weighted average of their ratings.

The prediction is given by:

$$
\hat{r}_{ui} =
\frac{
\sum_{v \in N(u)}
s(u,v)r_{vi}
}{
\sum_{v \in N(u)}
s(u,v)
}
$$

where:

- $u$ = target user
- $i$ = target movie
- $v$ = similar user
- $s(u,v)$ = similarity between users
- $r_{vi}$ = rating given by similar user $v$

The implementation uses:

```text
Number of neighbors (k) = 5
```

---

## 5.4 Evaluation

The model was evaluated using **Root Mean Squared Error (RMSE)**.

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

A lower RMSE indicates that the predicted ratings are closer to the
actual ratings.

### Result

**Memory-Based CF RMSE: 1.0906**

---

# 6. Model-Based Collaborative Filtering

## 6.1 Matrix Factorization

The second approach uses **Matrix Factorization**.

Instead of directly comparing users, the model learns latent representations
for users and movies.

The original rating matrix $R$ is approximated as:

$$
R \approx UV^T
$$

where:

- $U$ represents users using latent factors.
- $V$ represents movies using latent factors.

The model used **20 latent factors**.

Therefore:

$$
U \in \mathbb{R}^{610 \times 20}
$$

and:

$$
V \in \mathbb{R}^{8965 \times 20}
$$

---

## 6.2 Prediction

The predicted rating for a user-movie pair is calculated using the dot
product of their latent vectors:

$$
\hat{r}_{ui}=U_u\cdot V_i
$$

The user vector and movie vector are combined through their dot product
to produce the predicted rating.

---

# 7. Matrix Factorization Training

The user and movie latent-factor matrices were initialized using small
random values from a normal distribution.

No pretrained recommendation weights were used.

The model was trained using gradient-based updates.

For every training interaction:

$$
e_{ui}=r_{ui}-\hat{r}_{ui}
$$

where:

- $r_{ui}$ is the actual rating.
- $\hat{r}_{ui}$ is the predicted rating.
- $e_{ui}$ is the prediction error.

The latent vectors were updated using:

$$
U_u \leftarrow U_u+\alpha e_{ui}V_i
$$

$$
V_i \leftarrow V_i+\alpha e_{ui}U_u
$$

where $\alpha$ is the learning rate.

### Training Configuration

| Parameter | Value |
|---|---:|
| Latent Factors | 20 |
| Learning Rate | 0.01 |
| Epochs | 10 |

---

# 8. Training Results

The training Mean Squared Error decreased throughout the training process.

| Epoch | Training MSE |
|---:|---:|
| 1 | 10.1975 |
| 2 | 3.7560 |
| 3 | 1.8603 |
| 4 | 1.3019 |
| 5 | 1.0268 |
| 6 | 0.8609 |
| 7 | 0.7510 |
| 8 | 0.6727 |
| 9 | 0.6123 |
| 10 | 0.5625 |

The decrease in MSE indicates that the latent user and movie
representations were progressively adjusted to better fit the observed
training ratings.

---

# 9. Matrix Factorization Evaluation

The trained model was evaluated on the same test set used for the
memory-based approach.

### Result

**Matrix Factorization RMSE: 0.9524**

---

# 10. Model Comparison

| Model | Test RMSE |
|---|---:|
| Memory-Based Collaborative Filtering | 1.0906 |
| Matrix Factorization | **0.9524** |

Matrix Factorization achieved a lower test RMSE than the memory-based
approach.

The reduction in RMSE was approximately **13%** relative to the
memory-based baseline under this experimental setup.



---

# 11. Discussion

The two approaches solve the collaborative filtering problem in different
ways.

## Memory-Based Collaborative Filtering

Memory-based collaborative filtering directly uses relationships between
users.

It essentially asks:

> Which users behave similarly to this user?

### Advantages

- Simple to understand and implement.
- Easy to interpret.
- Directly uses observed user behavior.
- Does not require training a complex model.

### Limitations

- Depends on finding useful similar users.
- Performance can be affected by sparse interactions.
- Similarity calculations can become expensive as the number of users
  increases.
- The current implementation represents missing ratings as zero when
  calculating cosine similarity.

---

## Matrix Factorization

Matrix factorization attempts to discover hidden factors that explain the
observed user-item interactions.

It essentially asks:

> What latent factors can explain the preferences of users and the
> characteristics of items?

### Advantages

- Learns compact user and item representations.
- Can capture broader preference patterns.
- Works well with sparse interaction data.
- Achieved lower test RMSE in this experiment.

### Limitations

- Requires model training.
- Latent factors are less directly interpretable.
- Performance depends on choices such as the number of latent factors and
  learning rate.
- The current implementation does not include regularization or bias terms.

---

# 12. When to Use Each Approach

## Memory-Based Collaborative Filtering

Memory-based methods can be preferable when:

- Simplicity is important.
- Interpretability is important.
- The number of users is relatively manageable.
- Direct neighborhood-based recommendations are sufficient.

## Model-Based Collaborative Filtering

Model-based methods such as matrix factorization can be preferable when:

- The interaction matrix is large and sparse.
- Latent user and item preferences need to be learned.
- Better predictive performance is required.
- Compact representations of users and items are useful.

---

# 13. Conclusion

Both memory-based collaborative filtering and matrix factorization were
implemented using the same MovieLens dataset and the same per-user 80/20
train/test split.

The memory-based approach achieved:

$$
\boxed{RMSE = 1.0906}
$$

while matrix factorization achieved:

$$
\boxed{RMSE = 0.9524}
$$

Matrix factorization therefore performed better on this dataset and
reduced the test RMSE by approximately **13%** compared with the
memory-based baseline.

The experiment demonstrates the progression from a direct
similarity-based recommendation approach to a model that learns latent
representations of users and items.

However, the result does not imply that matrix factorization is always
superior. Memory-based methods remain useful when simplicity,
interpretability, and direct neighborhood relationships are important.
Model-based approaches become more attractive when learning latent
preferences from sparse interaction data is the priority.

---

# 14. Limitations

The current implementation has several limitations:

1. Missing ratings were filled with zero when calculating cosine
   similarity.

2. The memory-based model uses a fixed value of $k=5$ nearest neighbors.

3. Matrix factorization uses a fixed number of 20 latent factors.

4. The learning rate and number of epochs were not systematically tuned.

5. Regularization was not included in the matrix factorization model.

6. User and item bias terms were not included.

7. Movies that do not appear in the training matrix cannot be predicted by
   the current implementation.

These provide possible directions for future improvements.

---

# 15. Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**

---

# 16. Files

| File | Description |
|---|---|
| `README.md` | Task documentation |
| `recommender.py` | Memory-based collaborative filtering implementation |
| `matrix_factorization.py` | Matrix factorization implementation |
| `ratings.csv` | MovieLens ratings dataset |


---

# 17. Future Improvements

Possible improvements to the current implementation include:

- Experimenting with different values of $k$.
- Adding user and item bias terms.
- Adding L2 regularization to matrix factorization.
- Tuning the number of latent factors.
- Tuning the learning rate and number of epochs.
- Exploring alternative similarity calculations.
- Comparing additional collaborative filtering approaches.
- Evaluating recommendation ranking metrics such as Precision@K and
  Recall@K.
