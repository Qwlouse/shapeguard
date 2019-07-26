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
"""Defines the transformation from a shape template parse tree to ShapeSpec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shapeguard import dim_specs
from shapeguard import shape_spec
from shapeguard import shape_spec_parser


class TreeToSpec(shape_spec_parser.Transformer):
    start = shape_spec.ShapeSpec
    wildcard = dim_specs.Wildcard.make
    ellipsis = dim_specs.EllipsisDim.make
    dynamic = dim_specs.Dynamic.make
    name = dim_specs.NamedDim.make
    dynamic_name = dim_specs.DynamicNamedDim.make
    number = dim_specs.Number.make
    add = dim_specs.AddDims.make
    sub = dim_specs.SubDims.make
    mul = dim_specs.MulDims.make
    div = dim_specs.DivDims.make


parser = shape_spec_parser.Lark_StandAlone(transformer=TreeToSpec())
parse = parser.parse
