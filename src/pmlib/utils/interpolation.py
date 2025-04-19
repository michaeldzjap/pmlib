"""Interpolation utilities."""

from enum import Enum, auto
from math import floor
from typing import Self

import numpy as np


class InterpolationType(Enum):
    """All possible interpolation types."""

    LINEAR = auto()
    CUBIC = auto()


class SpreadingOperator:
    """
    Generate a distribution that can be used to spread a scalar value over a fixed
    size grid.

    Parameters
    ----------
    x : float
        The location about which to spread a scalar value.
    n : int
        The length of the resulting spreading distribution vector.
    interpolation_type : InterpolationType | None
        The type of interpolation that should be used for the spreading operation.
    """

    def __init__(
        self, x: float, n: int, interpolation_type: InterpolationType | None
    ) -> None:
        self.x = x
        self.n = n
        self.interpolation_type = interpolation_type
        self.__j = np.zeros((n, 1))

    @property
    def x(self) -> float:
        """Get the location about which to spread a scalar value."""
        return self.__x

    @x.setter
    def x(self, value: float):
        if not 0 <= value < 1:
            raise ValueError("x must be between 0 and 1")

        self.__x = value

    @property
    def n(self) -> int:
        """Get the length of the resulting spreading distribution vector."""
        return self.__n

    @n.setter
    def n(self, value: int):
        self.__n = value
        self.__h = 1 / value

    @property
    def interpolation_type(self) -> InterpolationType | None:
        """Get the interpolation type used for spreading out a scalar value."""
        return self.__interpolation_type

    @interpolation_type.setter
    def interpolation_type(self, value: InterpolationType | None):
        self.__interpolation_type = value

        if value is InterpolationType.LINEAR:
            self.__spreader = __linear_spread
            self.__clearer = __linear_clear
        elif value is InterpolationType.CUBIC:
            self.__spreader = __cubic_spread
            self.__clearer = __cubic_clear
        else:
            self.__spreader = __spread
            self.__clearer = __clear

    @property
    def distribution(self) -> np.ndarray:
        """Get the distribution vector."""
        return self.__j

    def build(self) -> Self:
        """
        Build the distribution.

        Returns
        -------
        SpreadingOperator
            The spreading operator instance.
        """
        self.__spreader(self.__x, self.__h, self.__j)

        return self

    def clear(self) -> Self:
        """
        Clear the distribution around the current spreading location.

        Returns
        -------
        SpreadingOperator
            The spreading operator instance.
        """
        self.__clearer(self.__x, self.__h, self.__j)

        return self

    def update(self, x: float) -> Self:
        """
        Clear and build the distribution.

        Parameters
        ----------
        x : float
            The location about which to spread a scalar value.

        Returns
        -------
        SpreadingOperator
            The spreading operator instance.
        """
        self.clear()

        self.x = x

        self.build()

        return self


def __spread(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))

    j[li] = 1

    return j


def __linear_spread(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))
    alpha = x / h - li
    indices = np.array([[li], [li + 1]])

    np.put(j, indices[indices < len(j)], [(1 - alpha), alpha])

    return j


def __cubic_spread(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))
    alpha = x / h - li
    indices = np.array([[li - 1], [li], [li + 1], [li + 2]])

    np.put(
        j,
        indices[np.all((indices >= 0, indices < len(j)), axis=0)],
        [
            alpha * (alpha - 1) * (alpha - 2) / -6,
            (alpha - 1) * (alpha + 1) * (alpha - 2) / 2,
            alpha * (alpha + 1) * (alpha - 2) / -2,
            alpha * (alpha + 1) * (alpha - 1) / 6,
        ],
    )

    return j


def __clear(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))

    j[li] = 0

    return j


def __linear_clear(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))
    indices = np.array([[li], [li + 1]])

    np.put(j, indices[indices < len(j)], 0)

    return j


def __cubic_clear(x: float, h: float, j: np.ndarray) -> np.ndarray:
    li = int(floor(x / h))
    indices = np.array([[li - 1], [li], [li + 1], [li + 2]])

    np.put(j, indices[np.all((indices >= 0, indices < len(j)), axis=0)], 0)

    return j
