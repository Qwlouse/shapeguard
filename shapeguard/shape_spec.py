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

"""Defines the ShapeSpec object which represents a parsed shape template."""

from shapeguard import dim_specs
from shapeguard import exception
from shapeguard import shape_spec_parser


class ShapeSpec(object):

    def __init__(self, entries):
        super(ShapeSpec, self).__init__()
        self.entries = [x for x in entries
                        if not isinstance(x, shape_spec_parser.Token)]
        if dim_specs.ellipsis_dim in self.entries:
            idx = self.entries.index(dim_specs.ellipsis_dim)
            self.left_entries = self.entries[:idx]
            self.right_entries = self.entries[idx + 1:]
            self.has_ellipsis = True
        else:
            self.left_entries = self.entries
            self.right_entries = []
            self.has_ellipsis = False

    def evaluate(self, known_dims=()):
        if self.has_ellipsis:
            raise exception.UnderspecifiedShapeError(
                "Template with an ellipsis (...) cannot be fully evaluated.")
        else:
            return [x.evaluate(known_dims) for x in self.entries]

    def partial_evaluate(self, known_dims=None):
        known_dims = known_dims or {}
        eshape = []
        for x in self.entries:
            try:
                eshape.append(x.evaluate(known_dims))
            except exception.UnderspecifiedShapeError:
                eshape.append(x)
        return eshape

    def rank_matches(self, shape):
        if self.has_ellipsis:
            if len(shape) < len(self.entries) - 1:
                return False
        else:
            if len(shape) != len(self.entries):
                return False
        return True

    def matches(self, shape, known_dims=()):
        rank_matches = self.rank_matches(shape)
        conflicts = any([x.has_conflict(s, known_dims)
                         for s, x in self.zip_iter(shape)])

        return rank_matches and not conflicts

    def zip_iter(self, shape):
        for s, e in zip(shape, self.left_entries):
            yield s, e
        if self.right_entries:
            for s, e in zip(shape[-len(self.right_entries):],
                            self.right_entries):
                yield s, e

    def infer(self, shape, known_dims=()):
        current_known = {}
        if known_dims:
            current_known.update(known_dims)
        inferred = True
        while inferred:
            inferred = {}
            for s, x in self.zip_iter(shape):
                inferred.update(x.infer(s, current_known))
            current_known.update(inferred)
        return current_known

    def __repr__(self):
        return "<{}>".format(self.entries)

    def __len__(self):
        return len(self.entries)
