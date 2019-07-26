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

import tensorflow as tf

from shapeguard.guard import ShapeGuard


def test_matches_basic_numerical():
    sg = ShapeGuard()
    a = tf.ones([1, 2, 3])
    assert sg.matches(a, "1, 2, 3")
    assert not sg.matches(a, "1, 2, 4")
    assert not sg.matches(a, "1, 2, 3, 4")
    assert not sg.matches(a, "1, 2")


def test_matches_ignores_spaces():
    sg = ShapeGuard()
    a = tf.ones([1, 2, 3])
    assert sg.matches(a, "1,2,3")
    assert sg.matches(a, "1 ,  2, 3   ")
    assert sg.matches(a, "1,  2,3 ")


def test_matches_named_dims():
    sg = ShapeGuard(dims={'N': 24, 'Z': 16})
    z = tf.ones([24, 16])
    assert sg.matches(z, "N, Z")
    assert sg.matches(z, "24, Z")
    assert not sg.matches(z, "N, N")


def test_matches_wildcards():
    sg = ShapeGuard()
    z = tf.ones([1, 2, 4, 8])
    assert sg.matches(z, "1, 2, 4, *")
    assert sg.matches(z, "*, *, *, 8")
    assert not sg.matches(z, "*")
    assert not sg.matches(z, "*, *, *")
