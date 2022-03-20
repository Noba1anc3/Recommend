# pip install -q --upgrade tensorflow-datasets

import pprint

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import numpy as np
import tensorflow as tf

ratings = tfds.load("movielens/100k-ratings", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

# I. Turning categorical features into embeddings
# I.I.I Define the vocabulary
movie_title_lookup = tf.keras.layers.StringLookup()
movie_title_lookup.adapt(ratings.map(lambda x: x["movie_title"]))
print(f"Vocabulary: {movie_title_lookup.get_vocabulary()[:3]}")
# Vocabulary: ['[UNK]', 'Star Wars (1977)', 'Contact (1997)']
movie_title_lookup(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])
# <tf.Tensor: shape=(2,), dtype=int64, numpy=array([ 1, 58])>

user_id_lookup = tf.keras.layers.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))

# I.I.II Using Feature Hashing
# We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 200_000
movie_title_hashing = tf.keras.layers.Hashing(num_bins=num_hashing_bins)
movie_title_hashing(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])
# <tf.Tensor: shape=(2,), dtype=int64, numpy=array([101016,  96565])>

# I.II Define the embeddings

user_id_embedding = tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32)
user_id_model = tf.keras.Sequential([user_id_lookup, user_id_embedding])

movie_title_embedding = tf.keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=movie_title_lookup.vocab_size(),
    output_dim=32
)
movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])

movie_title_model(["Star Wars (1977)"])
# <tf.Tensor: shape=(1, 32), dtype=float32, numpy=
# array([[-0.00255408,  0.00941082,  0.02599109, -0.02758816, -0.03652344,
#         -0.03852248, -0.03309812, -0.04343383,  0.03444691, -0.02454401,
#          0.00619583, -0.01912323, -0.03988413,  0.03595274,  0.00727529,
#          0.04844356,  0.04739804,  0.02836904,  0.01647964, -0.02924066,
#         -0.00425701,  0.01747661,  0.0114414 ,  0.04916174,  0.02185034,
#         -0.00399858,  0.03934855,  0.03666003,  0.01980535, -0.03694187,
#         -0.02149243, -0.03765338]], dtype=float32)>


# II. Normalizing continuous features

for x in ratings.take(3).as_numpy_iterator():
  print(f"Timestamp: {x['timestamp']}.")
# Timestamp: 879024327.
# Timestamp: 875654590.
# Timestamp: 882075110.

# II.I.I Standardization
timestamp_normalization = tf.keras.layers.Normalization(axis=None)
timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))
for x in ratings.take(3).as_numpy_iterator():
  print(f"Normalized timestamp: {timestamp_normalization(x['timestamp'])}.")
# Normalized timestamp: [-0.84293723].
# Normalized timestamp: [-1.4735204].
# Normalized timestamp: [-0.27203268].

# II.I.II Discretization
max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()
timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=1000)
print(f"Buckets: {timestamp_buckets[:3]}")
# Buckets: [8.74724710e+08 8.74743291e+08 8.74761871e+08]

# II.II
timestamp_embedding_model = tf.keras.Sequential([
  tf.keras.layers.Discretization(timestamp_buckets.tolist()),
  tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
])
for timestamp in ratings.take(1).map(lambda x: x["timestamp"]).batch(1).as_numpy_iterator():
  print(f"Timestamp embedding: {timestamp_embedding_model(timestamp)}.")
# Timestamp embedding: [[-0.02532113 -0.00415025  0.00458465  0.02080876  0.03103903 -0.03746337
#    0.04010465 -0.01709593 -0.00246077 -0.01220842  0.02456966 -0.04816503
#    0.04552222  0.03535838  0.00769508  0.04328252  0.00869263  0.01110227
#    0.02754457 -0.02659499 -0.01055292 -0.03035731  0.00463334 -0.02848787
#   -0.03416766  0.02538678 -0.03446608 -0.0384447  -0.03032914 -0.02391632
#    0.02637175 -0.01158618]].

# III. Text Features
title_text = tf.keras.layers.TextVectorization()
title_text.adapt(ratings.map(lambda x: x["movie_title"]))
for row in ratings.batch(1).map(lambda x: x["movie_title"]).take(1):
  print(title_text(row))
# tf.Tensor([[ 32 266 162   2 267 265  53]], shape=(1, 7), dtype=int64)



class UserModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        user_id_lookup,
        tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
    ])
    self.timestamp_embedding = tf.keras.Sequential([
      tf.keras.layers.Discretization(timestamp_buckets.tolist()),
      tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
    ])
    self.normalized_timestamp = tf.keras.layers.Normalization(
        axis=None
    )

  def call(self, inputs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
    ], axis=1)

user_model = UserModel()
user_model.normalized_timestamp.adapt(
    ratings.map(lambda x: x["timestamp"]).batch(128))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {user_model(row)[0, :3]}")
# Computed representations: [-0.04705765 -0.04739009 -0.04212048]

class MovieModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      movie_title_lookup,
      tf.keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
    ])
    self.title_text_embedding = tf.keras.Sequential([
      tf.keras.layers.TextVectorization(max_tokens=max_tokens),
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

  def call(self, inputs):
    return tf.concat([
        self.title_embedding(inputs["movie_title"]),
        self.title_text_embedding(inputs["movie_title"]),
    ], axis=1)


movie_model = MovieModel()
movie_model.title_text_embedding.layers[0].adapt(
    ratings.map(lambda x: x["movie_title"]))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {movie_model(row)[0, :3]}")
# Computed representations: [-0.01670959  0.02128791  0.04631067]

movies = tfds.load("movielens/1m-movies", split='train')
movies = movies.map(lambda x: x["movie_id"])


class MovielensModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      UserModel(),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      MovieModel(),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
      metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(self.candidate_model)
      )
    )
  
  def compute_loss(self, features, training=False):
    query_embeddings = self.query_model(features)
    movie_embeddings = self.candidate_model(features)

    return self.task(query_embeddings, movie_embeddings)