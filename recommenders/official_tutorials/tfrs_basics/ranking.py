# pip install -q tensorflow-recommenders
# pip install -q --upgrade tensorflow-datasets
# pip install -q scann

from cProfile import label
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class MovielensModel(tfrs.Model):
  def __init__(self, unique_user_ids, unique_movie_titles, task):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_movie_titles)
    self.task: tf.keras.layers.Layer = task

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    
    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)


class RankingModel(tf.keras.Model):
  def __init__(self, unique_user_ids, unique_movie_titles):
    super().__init__()
    embedding_dimension = 32
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=False),
      tf.keras.layers.Embedding(len(unique_user_ids)+1, embedding_dimension)
    ])
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=False),
      tf.keras.layers.Embedding(len(unique_movie_titles)+1, embedding_dimension)
    ])
    self.rating = tf.keras.Sequential([
      tf.keras.Dense(256, activation="relu"),
      tf.keras.Dense(64, activation="relu"),
      tf.keras.Dense(1)
    ])
  
  def __call__(self, inputs):
    user_id, movie_title = inputs
    
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)
    
    return self.rating(tf.concat([user_embedding, movie_embedding], axis=1))


# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# shuffle the dataset
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_titles"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# define metrics
metrics = tfrs.metrics.RootMeanSquaredError()

# define task and model
task = tfrs.tasks.Ranking(
  loss=tf.keras.losses.MeanSquaredError(),
  metrics=metrics)
model = MovielensModel(unique_user_ids, unique_movie_titles, task)

# train the model
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train, epochs=3)
model.evaluate(test, return_dict=True)

# prediction
test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({
      "user_id": np.array(["42"]),
      "movie_title": np.array([movie_title])
  })

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  # Save the index.
  tf.saved_model.save(model, path)
  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)
  # Pass a user id in, get top predicted movie titles back.
  scores = loaded({"user_id": np.array(["42"]), "movie_title": ["Speed (1994)"]}).numpy()
  print(f"Recommendations: {scores}")
