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

"""Defines all DimSpecs which represent individual dimensions of a ShapeSpec"""

import operator
from typing import Optional, Dict, Callable, TypeVar, Generic, Any, Union
from shapeguard import exception


# ############################################################################
# Hack to make mypy happy with callable class variables
# see https://github.com/python/mypy/issues/708
T = TypeVar("T")


class FunctionProperty(Generic[T]):
    def __get__(self, oself: Any, owner: Any) -> T:
        pass

    def __set__(self, oself: Any, value: T) -> None:
        pass


OperatorType = Callable[[Optional[int], Optional[int]], Optional[int]]
BinaryOperator = Union[OperatorType, FunctionProperty[OperatorType]]

# ############################################################################


class DimSpec:
    """Baseclass for single dimension specification."""

    @classmethod
    def make(cls, children=()):
        return cls(*children)

    def __init__(self, *args):
        super(DimSpec, self).__init__()

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        """Determines if this dim spec has a conflict with a given shape_entry.

        If there is not enough knowledge to decide, this function will assume
        no conflict and return False.

        Args:
          shape_entry: int or None. Dim-value of the shape to be checked
          known_dims: Dict[str, int]. Dictionary of known named dimension sizes

        Returns:
          True if there is a conflict (with given knowledge), False otherwise
        """
        raise NotImplementedError

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        """Evaluate the value of this dimension under the given known_dims.

        Args:
          known_dims: Dict[str, int]. Dictionary of known named dimension sizes

        Returns:
          int or None. The expected size of this dimension.

        Raises:
          UnderspecifiedShapeError: if there is not enough information to
            determine the size of the dimension.
        """
        raise NotImplementedError

    def infer(self,
              shape_entry: Optional[int],
              known_dims: Dict[str, int]) -> Dict[str, int]:
        """Try to infer named-dimension sizes from the given shape_entry.

        Args:
          shape_entry: int or None. Dim-value of the shape to be checked
          known_dims: Dict[str, int]. Dictionary of known named dimension sizes

        Returns:
          Dict[str, int]: dictionary of inferred named dimension sizes
        """
        return {}

    def flat_iter(self):
        """Iterate all multiplicative sub-components of this dimension."""
        yield self

    def __repr__(self) -> str:
        return '<DimSpec>'

    def __eq__(self, other) -> bool:
        return isinstance(other, DimSpec)


class EllipsisDim(DimSpec):
    """Represents zero or more wildcard dimensions."""

    instance = None

    @classmethod
    def make(cls, children=()):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        raise RuntimeError('Should never be called.')

    def __repr__(self) -> str:
        return '...'

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        raise exception.UnderspecifiedShapeError(
            'EllipsisDim cannot be evaluated.')

    def __eq__(self, other) -> bool:
        return isinstance(other, EllipsisDim)


ellipsis_dim = EllipsisDim.make()


class Wildcard(DimSpec):
    """Represents a dimension with any size."""

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        return False  # by definition never has a conflict

    def evaluate(self, known_dims) -> Optional[int]:
        return -1

    def __repr__(self) -> str:
        return '*'

    def __eq__(self, other) -> bool:
        return isinstance(other, Wildcard)


class Number(DimSpec):
    """Represents a dimension with a fixed numerical size."""

    def __init__(self, value: int):
        super(Number, self).__init__()
        self.value = int(value)

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        if shape_entry is None:
            return True
        else:
            return shape_entry != self.value

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        return self.value

    def __repr__(self) -> str:
        return '{}'.format(self.value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Number):
            return False
        else:
            return self.value == other.value


class Dynamic(DimSpec):
    """Represents a dynamic dimension (i.e. None entry in shape)."""

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        return shape_entry is not None  # only matches None dimensions

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        return None

    def __repr__(self):
        return 'None'

    def __eq__(self, other):
        return isinstance(other, Dynamic)


class NamedDim(DimSpec):
    """Represents a named dimension."""

    def __init__(self, name):
        super(NamedDim, self).__init__()
        self.name = str(name)

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        if shape_entry is None:
            return True
        elif self.name not in known_dims:
            return False
        else:
            return known_dims[self.name] != shape_entry

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        if self.name in known_dims:
            return known_dims[self.name]
        raise exception.UnderspecifiedShapeError(
            'Unknown dimension "{}"\nKnown dimensions: {}'.format(self.name,
                                                                  known_dims))

    def infer(self,
              shape_entry: Optional[int],
              known_dims: Dict[str, int]) -> Dict[str, int]:
        if shape_entry is None or self.name in known_dims:
            return {}
        else:
            return {self.name: shape_entry}

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, NamedDim):
            return False
        else:
            return self.name == other.name


class DynamicNamedDim(NamedDim):
    """Represents a dynamic or named dimension."""

    def __init__(self, name, _=None):
        super(DynamicNamedDim, self).__init__(name)

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        if shape_entry is None or self.name not in known_dims:
            return False
        else:
            return known_dims[self.name] != shape_entry

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        if self.name in known_dims:
            return known_dims[self.name]
        else:
            return None

    def __repr__(self):
        return self.name + '?'

    def __eq__(self, other):
        if not isinstance(other, NamedDim):
            return False
        else:
            return self.name == other.name


class OpSpec(DimSpec):
    """Baseclass for dimension operations."""

    op_str: str = '#'
    op: BinaryOperator
    left_op: BinaryOperator
    right_op: BinaryOperator

    def __init__(self, left, right):
        super(OpSpec, self).__init__()
        self.left: DimSpec = left
        self.right: DimSpec = right

    def evaluate(self, known_dims: Dict[str, int]) -> Optional[int]:
        return self.op(self.left.evaluate(known_dims),
                       self.right.evaluate(known_dims))

    def infer(self,
              shape_entry: Optional[int],
              known_dims: Dict[str, int]) -> Dict[str, int]:
        try:
            left_val = self.left.evaluate(known_dims)
            right_val = self.right_op(shape_entry, left_val)
            return self.right.infer(right_val, known_dims)
        except exception.UnderspecifiedShapeError:
            pass
        try:
            right_val = self.right.evaluate(known_dims)
            left_val = self.left_op(shape_entry, right_val)
            return self.left.infer(left_val, known_dims)
        except exception.UnderspecifiedShapeError:
            pass
        return {}

    def has_conflict(self,
                     shape_entry: Optional[int],
                     known_dims: Dict[str, int]) -> bool:
        if shape_entry is None:
            return False
        try:
            return self.evaluate(known_dims) != shape_entry
        except exception.UnderspecifiedShapeError:
            return False

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.left == other.left and self.right == other.right

    def __repr__(self):
        return '({} {} {})'.format(self.left, self.op_str, self.right)

    def flat_iter(self):
        if self.op == operator.mul:
            for c in self.left.flat_iter():
                yield c
            for c in self.right.flat_iter():
                yield c
        else:
            yield self


class AddDims(OpSpec):
    """Represents addition of two dimension values."""

    op_str = '+'
    op = operator.add
    left_op = operator.sub
    right_op = operator.sub


class SubDims(OpSpec):
    """Represents subtraction of two dimension values."""

    op_str = '-'
    op = operator.sub
    left_op = operator.add
    right_op = operator.sub


class MulDims(OpSpec):
    """Represents product of two dimension values."""

    op_str = '*'
    op = operator.mul
    left_op = operator.floordiv
    right_op = operator.floordiv


class DivDims(OpSpec):
    """Represents quotient of two dimension values."""

    op_str = '/'
    op = operator.floordiv
    left_op = operator.mul
    right_op = operator.floordiv
