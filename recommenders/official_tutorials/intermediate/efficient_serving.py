# pip install -q scann

from typing import Dict, Text

import os
import pprint
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Load the MovieLens 100K data.
ratings = tfds.load(
    "movielens/100k-ratings",
    split="train"
)

# Get the ratings data.
ratings = (ratings
           # Retain only the fields we need.
           .map(lambda x: {"user_id": x["user_id"], "movie_title": x["movie_title"]})
           # Cache for efficiency.
           .cache(tempfile.NamedTemporaryFile().name)
)

# Get the movies data.
movies = tfds.load("movielens/100k-movies", split="train")
movies = (movies
          # Retain only the fields we need.
          .map(lambda x: x["movie_title"])
          # Cache for efficiency.
          .cache(tempfile.NamedTemporaryFile().name))

user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(user_ids.batch(1000))))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)


class MovielensModel(tfrs.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32

    # Set up a model for representing movies.
    self.movie_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Set up a model for representing users.
    self.user_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Set up a task to optimize the model and compute metrics.
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).cache().map(self.movie_model)
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.

    return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=not training)


model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train.batch(8192), epochs=3)
model.evaluate(test.batch(8192), return_dict=True)

brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
brute_force.index_from_dataset(
    movies.batch(128).map(lambda title: (title, model.movie_model(title)))
)

# Get predictions for user 42.
_, titles = brute_force(np.array(["42"]), k=3)
print(f"Top recommendations: {titles[0]}")
# Top recommendations: [b'Homeward Bound: The Incredible Journey (1993)'
#  b"Kid in King Arthur's Court, A (1995)" b'Rudy (1993)']

# %timeit _, titles = brute_force(np.array(["42"]), k=3)
# 983 µs ± 5.44 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# Construct a dataset of movies that's 1,000 times larger. We 
# do this by adding several million dummy movie titles to the dataset.
lots_of_movies = tf.data.Dataset.concatenate(
    movies.batch(4096),
    movies.batch(4096).repeat(1_000).map(lambda x: tf.zeros_like(x))
)

# We also add lots of dummy embeddings by randomly perturbing
# the estimated embeddings for real movies.
lots_of_movies_embeddings = tf.data.Dataset.concatenate(
    movies.batch(4096).map(model.movie_model),
    movies.batch(4096).repeat(1_000)
      .map(lambda x: model.movie_model(x))
      .map(lambda x: x * tf.random.uniform(tf.shape(x)))
)

brute_force_lots = tfrs.layers.factorized_top_k.BruteForce()
brute_force_lots.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)

_, titles = brute_force_lots(model.user_model(np.array(["42"])), k=3)
print(f"Top recommendations: {titles[0]}")
# Top recommendations: [b'Homeward Bound: The Incredible Journey (1993)'
#  b"Kid in King Arthur's Court, A (1995)" b'Rudy (1993)']

# %timeit _, titles = brute_force_lots(model.user_model(np.array(["42"])), k=3)
# 33 ms ± 245 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

scann = tfrs.layers.factorized_top_k.ScaNN(num_reordering_candidates=100)
scann.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)

_, titles = scann(model.user_model(np.array(["42"])), k=3)
print(f"Top recommendations: {titles[0]}")
# Top recommendations: [b'Homeward Bound: The Incredible Journey (1993)'
#  b"Kid in King Arthur's Court, A (1995)" b'Rudy (1993)']

# %timeit _, titles = scann(model.user_model(np.array(["42"])), k=3)
# 4.35 ms ± 34.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)




# Override the existing streaming candidate source.
model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
    candidates=lots_of_movies_embeddings
)
# Need to recompile the model for the changes to take effect.
model.compile()

# %time baseline_result = model.evaluate(test.batch(8192), return_dict=True, verbose=False)
# CPU times: user 22min 5s, sys: 2min 7s, total: 24min 12s
# Wall time: 51.9 s

model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
    candidates=scann
)
model.compile()

# We can use a much bigger batch size here because ScaNN evaluation
# is more memory efficient.
# %time scann_result = model.evaluate(test.batch(8192), return_dict=True, verbose=False)
# CPU times: user 10.5 s, sys: 3.26 s, total: 13.7 s
# Wall time: 1.85 s

print(f"Brute force top-100 accuracy: \
    {baseline_result['factorized_top_k/top_100_categorical_accuracy']:.2f}")
print(f"ScaNN top-100 accuracy:       \
    {scann_result['factorized_top_k/top_100_categorical_accuracy']:.2f}")
# Brute force top-100 accuracy: 0.15
# ScaNN top-100 accuracy:       0.27



# We re-index the ScaNN layer to include the user embeddings in the same model.
# This way we can give the saved model raw features and get valid predictions back.
scann = tfrs.layers.factorized_top_k.ScaNN(model.user_model, num_reordering_candidates=1000)
scann.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)
# Need to call it to set the shapes.
_ = scann(np.array(["42"]))

with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  tf.saved_model.save(
      scann,
      path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  loaded = tf.saved_model.load(path)

_, titles = loaded(tf.constant(["42"]))
print(f"Top recommendations: {titles[0][:3]}")
# Top recommendations: [b'Homeward Bound: The Incredible Journey (1993)'
#  b"Kid in King Arthur's Court, A (1995)" b'Rudy (1993)']



# Process queries in groups of 1000; processing them all at once with brute force
# may lead to out-of-memory errors, because processing a batch of q queries against
# a size-n dataset takes O(nq) space with brute force.
titles_ground_truth = tf.concat([
  brute_force_lots(queries, k=10)[1] for queries in
  test.batch(1000).map(lambda x: model.user_model(x["user_id"]))
], axis=0)

# Get all user_id's as a 1d tensor of strings
test_flat = np.concatenate(list(test.map(lambda x: x["user_id"]).batch(1000).
                                  as_numpy_iterator()), axis=0)

# ScaNN is much more memory efficient and has no problem processing the whole
# batch of 20000 queries at once.
_, titles = scann(test_flat, k=10)

def compute_recall(ground_truth, approx_results):
  return np.mean([
      len(np.intersect1d(truth, approx)) / len(truth)
      for truth, approx in zip(ground_truth, approx_results)
  ])

print(f"Recall: {compute_recall(titles_ground_truth, titles):.3f}")
# Recall: 0.931
# %timeit -n 1000 scann(np.array(["42"]), k=10)
# 4.67 ms ± 25 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

scann2 = tfrs.layers.factorized_top_k.ScaNN(
    model.user_model, 
    num_leaves=1000,
    num_leaves_to_search=100,
    num_reordering_candidates=1000)
scann2.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)

_, titles2 = scann2(test_flat, k=10)
print(f"Recall: {compute_recall(titles_ground_truth, titles2):.3f}")
# Recall: 0.966
# %timeit -n 1000 scann2(np.array(["42"]), k=10)
# 4.86 ms ± 21.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

scann3 = tfrs.layers.factorized_top_k.ScaNN(
    model.user_model,
    num_leaves=1000,
    num_leaves_to_search=70,
    num_reordering_candidates=400)
scann3.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)

_, titles3 = scann3(test_flat, k=10)
print(f"Recall: {compute_recall(titles_ground_truth, titles3):.3f}")
# Recall: 0.957
# %timeit -n 1000 scann3(np.array(["42"]), k=10)
# 4.58 ms ± 37.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
