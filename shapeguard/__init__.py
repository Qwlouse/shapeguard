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

"""This python module contains ShapeGuard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapeguard.exception import ShapeError
from shapeguard.__about__ import __version__
from shapeguard.__about__ import __author__
from shapeguard.__about__ import __author_email__
from shapeguard.guard import ShapeGuard
from shapeguard.tools import matches
from shapeguard.tools import evaluate
from shapeguard.tools import reshape
from shapeguard.tools import get_shape


__all__ = ('ShapeGuard', '__version__', '__author__', '__author_email__',
           'matches', 'evaluate', 'guard', 'reshape', 'get_shape',
           'ShapeError')
