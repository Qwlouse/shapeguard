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

"""This module contains meta-information about the ShapeGuard package.

It is kept simple and separate from the main module, because this information
is also read by the setup.py, and during installation the shapeguard module
cannot yet be imported.
"""

__all__ = ('__version__', '__author__', '__author_email__')

__version__ = '0.1'

__author__ = 'Klaus Greff'
__author_email__ = 'klaus.greff@startmail.com'

__url__ = 'https://github.com/Qwlouse/shapeguard'
