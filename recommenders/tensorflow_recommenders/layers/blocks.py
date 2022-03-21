"""Convenience blocks for building models."""

from typing import List, Optional

import tensorflow as tf

from tensorflow_recommenders import types


class MLP(tf.keras.layers.Layer):
  """Sequential multi-layer perceptron (MLP) block."""

  def __init__(
      self,
      units: List[int],
      use_bias: bool = True,
      activation: Optional[types.Activation] = "relu",
      final_activation: Optional[types.Activation] = None,
      **kwargs) -> None:
    """Initializes the MLP layer.

    Args:
      units: Sequential list of layer sizes.
      use_bias: Whether to include a bias term.
      activation: Type of activation to use on all except the last layer.
      final_activation: Type of activation to use on last layer.
      **kwargs: Extra args passed to the Keras Layer base class.
    """

    super().__init__(**kwargs)

    self._sublayers = []

    for num_units in units[:-1]:
      self._sublayers.append(
          tf.keras.layers.Dense(
              num_units, activation=activation, use_bias=use_bias))
    self._sublayers.append(
        tf.keras.layers.Dense(
            units[-1], activation=final_activation, use_bias=use_bias))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Performs the forward computation of the block."""
    for layer in self._sublayers:
      x = layer(x)

    return x
