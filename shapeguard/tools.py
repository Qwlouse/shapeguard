# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the main ShapeGuard class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from shapeguard import exception
from shapeguard import parser


def matches(tensor, template, dims):
  shape = get_shape(tensor)
  spec = parser.parse(template)
  return spec.matches(shape, dims)


def reshape(tensor, template, dims):
  spec = parser.parse(template)
  new_shape = spec.evaluate(dims)
  return tf.reshape(tensor, new_shape)


def evaluate(template, dims):
  template = parser.parse(template)
  return template.evaluate(dims)


def guard(tensor, template, dims):
  shape = get_shape(tensor)
  spec = parser.parse(template)
  # compare rank
  if not spec.rank_matches(shape):
    raise exception.ShapeError(
        'Tensor has the wrong rank ({} != {}).\n'
        'Expected shape: {} (from template {})\n'
        '  Actual shape: {}'.format(
            len(shape), len(spec),
            spec.partial_evaluate(dims), template,
            shape))
  # infer dimensions
  inferred_dims = spec.infer(shape, dims)
  known_dims = copy(dims)
  known_dims.update(inferred_dims)
  # check if dimensions match
  if not spec.matches(shape, known_dims):
    raise exception.ShapeError(
        'Shape Mismatch\n'
        'Expected shape: {} (from template {})\n'
        '  Actual shape: {}'.format(
            spec.partial_evaluate(dims), template,
            shape))

  # return the inferred dims unless they start with '_'
  return {k: v for k, v in inferred_dims.items()
          if not k.startswith('_')}


def get_shape(tensor_or_shape):
  if isinstance(tensor_or_shape, (list, tuple)):
    return list(tensor_or_shape)
  elif isinstance(tensor_or_shape, tf.Tensor):
    return tensor_or_shape.get_shape().as_list()
  elif isinstance(tensor_or_shape, tf.TensorShape):
    return tensor_or_shape.as_list()
  elif isinstance(tensor_or_shape, np.ndarray):
    return tensor_or_shape.shape
  elif isinstance(tensor_or_shape, tfp.distributions.Distribution):
    return (tensor_or_shape.batch_shape.as_list() +
            tensor_or_shape.event_shape.as_list())
  else:
    raise TypeError('Unknown tensor/shape {} of type: {}'
                    .format(tensor_or_shape, type(tensor_or_shape)))
