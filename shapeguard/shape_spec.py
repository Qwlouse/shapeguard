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

from typing import List, Union, Dict, Optional, Tuple

from shapeguard import dim_specs
from shapeguard import exception
from shapeguard import shape_spec_parser

EntriesType = List[Union[shape_spec_parser.Token, dim_specs.DimSpec]]
ShapeType = Union[Tuple[int], List[int]]


class ShapeSpec:

    def __init__(self, entries: EntriesType):
        super().__init__()
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

    def evaluate(self,
                 known_dims: Dict[str, int] = None) -> List[Optional[int]]:
        known_dims = known_dims or {}

        if self.has_ellipsis:
            raise exception.UnderspecifiedShapeError(
                "Template with an ellipsis (...) cannot be fully evaluated.")
        else:
            return [x.evaluate(known_dims) for x in self.entries]

    def partial_evaluate(self,
                         known_dims: Dict[str, int] = None
                         ) -> List[Union[int, str, None]]:
        known_dims = known_dims or {}
        eval_shape: List[Union[int, str, None]] = []
        for x in self.entries:
            try:
                eval_shape.append(x.evaluate(known_dims))
            except exception.UnderspecifiedShapeError:
                eval_shape.append(repr(x))
        return eval_shape

    def rank_matches(self, shape: ShapeType) -> bool:
        if self.has_ellipsis:
            if len(shape) < len(self.entries) - 1:
                return False
        else:
            if len(shape) != len(self.entries):
                return False
        return True

    def matches(self, shape, known_dims: Dict[str, int] = None) -> bool:
        known_dims = known_dims or {}
        rank_matches = self.rank_matches(shape)
        conflicts = any([x.has_conflict(s, known_dims)
                         for s, x in self.zip_iter(shape)])

        return rank_matches and not conflicts

    def zip_iter(self, shape: ShapeType):
        for s, e in zip(shape, self.left_entries):
            yield s, e
        if self.right_entries:
            for s, e in zip(shape[-len(self.right_entries):],
                            self.right_entries):
                yield s, e

    def infer(self,
              shape: ShapeType,
              known_dims: Dict[str, int] = None) -> Dict[str, int]:
        current_known = {}
        if known_dims:
            current_known.update(known_dims)
        inferred = {'Start': True}
        while inferred:
            inferred = {}
            for s, x in self.zip_iter(shape):
                inferred.update(x.infer(s, current_known))
            current_known.update(inferred)
        return current_known

    def __repr__(self) -> str:
        return "<{}>".format(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
