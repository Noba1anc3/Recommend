## examples

```python
from tensorflow_recommenders.examples import movielens
```

```python
np.concatenate()
tf.data.Dataset.batch(batch_size)
tf.data.Dataset.map(lambda x: x["movie_id"])
ndarray.tolist()
collections.defaultdict()
@ # matrix multiplication
np.from_buffer(Array)
np.argsort(Array) # get index of Array to sort in ascending way
np.mean()
for key, value in dict.items():
np.random.RandomState().choice(range, size)
```

## models

```python
from tensorflow_recommenders.models import Model
```

```python
with tf.GradientTape() as tape:
    loss = self.compute_loss(inputs, training=True)
gradients = tape.gradient(total_loss, self.trainable_variables)
self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))    



```

### base

```python
class Model(tf.keras.Model):
	def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
    def call(self, inputs):
```



### base_test

```python
tf.keras.layers.Dense(units, activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None,)

tf.keras.losses.BinaryCrossentropy()
tf.keras.metrics.BinaryAccuracy(name="accuracy")
tf.keras.metrics.Mean(name="loss_mean")

# configures the model for training
tf.keras.Model.compile(optimizer, loss, metrics, ...)

tf.keras.Model.fit(x, y, batch_size, epochs, verbose, callbacks,
                   validation_split, valodation_data, shuffle,
                   class_weight, sample_weight, initial_epoch,
                   ...)

```



## tfrs.tasks

```python
basefrom tensorflow_recommenders.tasks import base
```

### ranking

```python
tfrs.tasks.Ranking(
    loss: Optional[tf.keras.losses.Loss] = None,
    metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    name: Optional[Text] = None
) -> None
```

```python
call(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    sample_weight: Optional[tf.Tensor] = None,
    training: bool = False,
    compute_metrics: bool = True
) -> tf.Tensor
```

```python
self._ranking_metrics = metrics or []
tf.keras.losses.Loss
Loss.call(y_true, y_pred, sample_weight)
tf.identity(Tensor) # return a tensor with the same shape and contents as input
```

### ranking_test

```python
tf.test.TestCase.assertAllClose(a, b)

with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
        loss = task(predictions=predictions, labels=labels)
        sess.run([var.initializer for var in task.variables])
        for metric in task.metrics:
          sess.run([var.initializer for var in metric.variables])
        sess.run(loss)
```

### retrieval

```python
tfrs.tasks.Retrieval(
    loss: Optional[tf.keras.losses.Loss] = None,
    metrics: Optional[tfrs.metrics.FactorizedTopK] = None,
    batch_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    temperature: Optional[float] = None,
    num_hard_negatives: Optional[int] = None,
    name: Optional[Text] = None
) -> None
```

```python
call(
    query_embeddings: tf.Tensor,
    candidate_embeddings: tf.Tensor,
    sample_weight: Optional[tf.Tensor] = None,
    candidate_sampling_probability: Optional[tf.Tensor] = None,
    candidate_ids: Optional[tf.Tensor] = None,
    compute_metrics: bool = True
) -> tf.Tensor
```

```python
tf.keras.losses.CategoricalCrossentropy()
@property
@factorized_metrics.setter
tf.linalg.matmul(a, b, transpose_a=False, transpose_b=True)
```

### retrieval_test

```python

```



## layers

### MLP

```python
tf.keras.layers.Dense(num_units, activation=activation, use_bias=use_bias)
```

### Cross

```python
tf.keras.initializers.Initializer
tf.keras.regularizers.Regularizer
tf.keras.initializers.get(kernel_initializer)
tf.keras.regularizers.get(kernel_regularizer)
list(Dict.items()) # Dict -> List

# dcn_test
np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
np.random.uniform(size=(10, 13))
tf.keras.models.load_model(path)
model.predict(random_input)
```

### DocInteraction

```python
xactions = tf.matmul(concat_features, concat_features, transpose_b=True)

# 以对角线为中心，取它的副对角线部分，其他部分用0填充
# num_lower:下三角矩阵保留的副对角线数量，取值为负数时，则全部保留。
# num_upper:上三角矩阵保留的副对角线数量，取值为负数时，则全部保留。
lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
upper_tri_mask = tf.linalg.band_part(ones, 0, -1)

activations = tf.boolean_mask(xactions, lower_tri_mask)
activations = tf.where(condition=tf.cast(upper_tri_mask, tf.bool),x=tf.zeros_like(xactions),y=xactions)
```

## loss

### SamplingProbabilityCorrection

```python
tf.clip_by_value(candidate_sampling_probability, 1e-6, 1.)
```

### RemoveAccidentalHits

```python
candidate_ids = tf.expand_dims(candidate_ids, 1)
positive_indices = tf.math.argmax(labels, axis=1)
```



## tutorials

### intermediate

#### dcn

```python
rng = np.random.RandomState(random_seed)
country = rng.randint(200, size=[data_size, 1]) / 200.
```

#### efficient_serving

```python
np.intersect1d(truth, approx)
```

#### listwise_ranking

```python
user_embedding_repeated = tf.repeat(
    tf.expand_dims(user_embeddings, 1), [list_length], axis=1)
predictions=tf.squeeze(scores, axis=-1)
tf.keras.losses.MeanSquaredError()
tfr.keras.losses.PairwiseHingeLoss()
tfr.keras.losses.ListMLELoss()
tf.keras.metrics.RootMeanSquaredError()
tfr.keras.metrics.NDCGMetric(name="ndcg_metric")
```



