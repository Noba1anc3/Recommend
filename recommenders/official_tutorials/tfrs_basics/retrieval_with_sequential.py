# pip install -q tensorflow-recommenders
# pip install -q --upgrade tensorflow-datasets
# pip install -q scann

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# wget -nc https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/recommendation/ml/data/example_generation_movielens.py
# python -m example_generation_movielens  --data_dir=data/raw  --output_dir=data/examples  --min_timeline_length=3  --max_context_length=10  --max_context_movie_genre_length=10  --min_rating=2  --train_data_fraction=0.9  --build_vocabs=False

class MovielensModel(tfrs.Model):
  def __init__(self, query_model, candidate_model, task):
    super().__init__()
    self._query_model: tf.keras.Model = query_model
    self._candidate_model: tf.keras.Model = candidate_model
    self._task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    watch_history = features["context_movie_id"]
    watch_next_label = features["label_movie_id"]

    query_embeddings = self._query_model(watch_history)
    candidate_embeddings = self._candidate_model(watch_next_label)

    # The task computes the loss and the metrics.
    return self._task(query_embeddings, candidate_embeddings, compute_metrics = not training)


train_filename = "./data/examples/train_movielens_1m.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = "./data/examples/test_movielens_1m.tfrecord"
test = tf.data.TFRecordDataset(test_filename)

feature_description = {
    'context_movie_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_movie_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
    'context_movie_year': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(1980, 10)),
    'context_movie_genre': tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("Drama", 10)),
    'label_movie_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

train_ds = train.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

for x in train_ds.take(1).as_numpy_iterator():
  pprint.pprint(x)

movies = tfds.load("movielens/1m-movies", split='train')
movies = movies.map(lambda x: x["movie_id"])
movie_ids = movies.batch(1_000)
unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

# define the model
embedding_dimension = 32
query_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(vocabulary=unique_movie_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_movie_ids)+1, embedding_dimension),
  tf.keras.layers.GRU(embedding_dimension)])
candidate_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(vocabulary=unique_movie_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_ids)+1, embedding_dimension)])

# define metrics
metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(candidate_model))

# define task and model
task = tfrs.tasks.Retrieval(metrics=metrics)
model = MovielensModel(query_model, candidate_model, task)

# train the model
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train, epochs=3)
model.evaluate(test, return_dict=True)

# prediction
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
    tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model))))
# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  # Save the index.
  tf.saved_model.save(index, path)
  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)
  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])
  print(f"Recommendations: {titles[0][:3]}")

# ScaNN
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model))))
# Get recommendations.
_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  # Save the index.
  tf.saved_model.save(index, path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)
  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])
  print(f"Recommendations: {titles[0][:3]}")