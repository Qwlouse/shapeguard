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

from copy import copy

from shapeguard import tools


class ShapeGuard(object):

    def __init__(self, dims=None):
        object.__setattr__(self, 'dims', {} if dims is None else dims)

    def matches(self, tensor, template):
        return tools.matches(tensor, template, self.dims)

    def guard(self, tensor, template):
        inferred_dims = tools.guard(tensor, template, self.dims)
        self.dims.update(inferred_dims)
        return tensor

    def reshape(self, tensor, template):
        return tools.reshape(tensor, template, self.dims)

    def evaluate(self, template, **kwargs):
        local_dims = copy(self.dims)
        local_dims.update(kwargs)
        return tools.evaluate(template, local_dims)

    def __getitem__(self, item):
        return tools.evaluate(item, self.dims)

    def __getattr__(self, item):
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return self.dims[item]
            except KeyError:
                raise AttributeError(item)

    def __setattr__(self, key, value):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, key)
        except AttributeError:
            try:
                self.dims[key] = value
            except KeyError:
                raise AttributeError(key)
        else:
            object.__setattr__(self, key, value)

    def __delattr__(self, item):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, item)
        except AttributeError:
            try:
                del self.dims[item]
            except KeyError:
                raise AttributeError(item)
        else:
            object.__delattr__(self, item)
