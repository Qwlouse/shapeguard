# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tensorflow as tf
from shapeguard import ShapeError
from shapeguard import ShapeGuard


def test_guard_raises():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  with pytest.raises(ShapeError):
    sg.guard(a, "3, 2, 1")


def test_guard_infers_dimensions():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  sg.guard(a, "A, B, C")
  assert sg.dims == {'A': 1, 'B': 2, 'C': 3}


def test_guard_infers_dimensions_complex():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  sg.guard(a, "A, B*2, A+C")
  assert sg.dims == {'A': 1, 'B': 1, 'C': 2}


def test_guard_infers_dimensions_operator_priority():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 8])
  sg.guard(a, "A, B, A+C*2+1")
  assert sg.dims == {'A': 1, 'B': 2, 'C': 3}


def test_guard_raises_complex():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  with pytest.raises(ShapeError):
    sg.guard(a, "A, B, B")


def test_guard_raises_inferred():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  b = tf.ones([3, 2, 5])
  sg.guard(a, "A, B, C")
  with pytest.raises(ShapeError):
    sg.guard(b, "C, B, A")


def test_guard_ignores_wildcard():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3])
  sg.guard(a, "*, *, 3")
  assert sg.dims == {}


def test_guard_dynamic_shape():
  sg = ShapeGuard()
  with pytest.raises(ShapeError):
    sg.guard([None, 2, 3], "C, B, A")

  sg.guard([None, 2, 3], "?, B, A")
  sg.guard([1, 2, 3], "C?, B, A")
  sg.guard([None, 2, 3], "C?, B, A")


def test_guard_ellipsis():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3, 4, 5])
  sg.guard(a, "...")
  sg.guard(a, "..., 5")
  sg.guard(a, "..., 4, 5")
  sg.guard(a, "1, ...")
  sg.guard(a, "1, 2, ...")
  sg.guard(a, "1, 2, ..., 4, 5")
  sg.guard(a, "1, 2, 3, ..., 4, 5")

  with pytest.raises(ShapeError):
    sg.guard(a, "1, 2, 3, 4, 5, 6,...")

  with pytest.raises(ShapeError):
    sg.guard(a, "..., 1, 2, 3, 4, 5, 6")


def test_guard_ellipsis_infer_dims():
  sg = ShapeGuard()
  a = tf.ones([1, 2, 3, 4, 5])
  sg.guard(a, "A, B, ..., C")
  assert sg.dims == {'A': 1, 'B': 2, 'C': 5}
