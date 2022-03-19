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


