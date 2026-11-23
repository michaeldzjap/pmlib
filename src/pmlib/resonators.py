"""Resonator definitions."""

from abc import ABC
from enum import Enum, auto
from inspect import isabstract
from typing import Any


class BoundaryCondition(Enum):
    """All supported boundary conditions."""

    CLAMPED = auto()
    SIMPLY_SUPPORTED = auto()


class Resonator(ABC):
    """
    Abstract base class from which all concrete resonator classes inherit.
    """

    _sr = 44100
    _k = 1.0 / _sr

    DIMENSION: int

    def __init__(
        self,
        gamma: float = 200.0,
        kappa: float = 1.0,
        sigma0: float = 1.0,
        sigma1: float = 1.0,
        M: float = 0.01,
    ):
        if gamma < 0.0 or kappa < 0.0 or (gamma == 0.0 and kappa == 0.0):
            raise ValueError(
                "Gamma and kappa must be non-negative and at least one must be greater than 0."
            )

        if sigma0 < 0.0 or sigma1 < 0.0:
            raise ValueError(
                "Damping coefficients sigma0 and sigma1 must be non-negative."
            )

        if M <= 0.0:
            raise ValueError("Mass M must be greater than zero.")

        self.gamma = gamma
        self.kappa = kappa
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.M = M

    def __init__subclass__(self, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if isabstract(self):  # Skip the check if this subclass is still abstract
            return
        if not hasattr(self, "DIMENSION"):
            raise TypeError(f"{self.__name__} must define class attribute 'DIMENSION'")

    @classmethod
    def sample_rate(cls, value: int) -> None:
        """
        Set the sample rate.

        Parameters
        ----------
        value : int
            The new sample rate.

        Raises
        ------
        ValueError
            If `value` is less than or equal to zero.
        """
        if value <= 0:
            raise ValueError("Sample rate must be greater than zero.")

        cls._sr = value
        cls._k = 1.0 / value
