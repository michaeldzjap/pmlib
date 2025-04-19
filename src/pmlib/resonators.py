"""Resonator definitions."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from math import sqrt
from typing import Self

from scipy.sparse import csr_matrix, dok_matrix, identity, spmatrix

from .utils.matrices import fourth_difference_matrix, second_difference_matrix


class BoundaryCondition(Enum):
    """All possible boundary conditions."""

    CLAMPED = auto()
    SIMPLY_SUPPORTED = auto()
    FREE = auto()


class Endpoint(Enum):
    """All possible endpoints."""

    LEFT = auto()
    RIGHT = auto()


class Resonator(ABC):
    """
    Abstract base class from which all concrete resonator classes inherit.
    """

    MIN_SAMPLE_RATE = 441
    MAX_SAMPLE_RATE = 705600

    _sample_rate = 44100
    _nyquist = _sample_rate / 2
    _time_step = 1 / _sample_rate

    def __init__(self, gamma: float, kappa: float, loss: tuple[float, float]):
        self.gamma = gamma
        self.kappa = kappa
        self.loss = loss
        self._b = csr_matrix((0, 0))
        self._c = csr_matrix((0, 0))

    @property
    def gamma(self) -> float:
        """Get the spatially scaled wave speed."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if not 0 <= value <= self._nyquist:
            raise ValueError(f"gamma must be between 0 and {self._nyquist}")

        self._gamma: float = value

    @property
    def kappa(self) -> float:
        """Get the spatially scaled stiffness coefficient."""
        return self._kappa

    @kappa.setter
    def kappa(self, value: float):
        if value < 0:
            raise ValueError("kappa must be equal or greater than 0")

        self._kappa = value

    @property
    def loss(self) -> tuple[float, float]:
        """Get the frequency independent and dependent damping constants."""
        return self._loss

    @loss.setter
    def loss(self, value: tuple[float, float]):
        for i, k in enumerate(value):
            if k < 0:
                raise ValueError(f"loss[{i}] must be equal or greater than 0")

        self._loss = value

    @property
    def b(self) -> spmatrix:
        """Get the update matrix acting on u^n."""
        return self._b

    @property
    def c(self) -> spmatrix:
        """Get the update matrix acting on u^(n - 1)."""
        return self._c

    @classmethod
    def sample_rate(cls, value: int) -> None:
        """
        Set the sample rate.

        Attributes
        ----------
        value : int
            The new sample rate.

        Raises
        ------
        ValueError
            If `value` is smaller than the minimum allowed or greater than the maximum allowed sample rate.
        """
        if not cls.MIN_SAMPLE_RATE <= value <= cls.MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample rate must be between {cls.MIN_SAMPLE_RATE} and {cls.MAX_SAMPLE_RATE}"
            )

        cls._sample_rate = value
        cls._nyquist = value / 2
        cls._time_step = 1 / value

    @abstractmethod
    def build(self) -> Self:
        """Build the resonator model."""

    @abstractmethod
    def _compute_grid_step(self) -> tuple[float, int]:
        pass


class LinearResonator1D(Resonator):
    """
    A concrete implementation of a one dimenstional linear resonator.

    Parameters
    ----------
    gamma : float, default 200
        The spatially scaled wave speed.
    kappa : float, default 1
        The spatially scaled stiffness coefficient.
    loss : tuple, default (0, 0)
        The frequency independent and dependent damping constants.
    boundary_conditions : dict, default {
                                            Endpoint.LEFT: BoundaryCondition.SIMPLY_SUPPORTED,
                                            Endpoint.RIGHT: BoundaryCondition.SIMPLY_SUPPORTED,
                                        }
        The left and right boundary conditions.
    """

    def __init__(
        self,
        gamma: float = 200,
        kappa: float = 1,
        loss: tuple[float, float] = (0, 0),
        boundary_conditions: dict[Endpoint, BoundaryCondition] | None = None,
    ):
        super().__init__(gamma, kappa, loss)

        if boundary_conditions is None:
            boundary_conditions = {
                Endpoint.LEFT: BoundaryCondition.SIMPLY_SUPPORTED,
                Endpoint.RIGHT: BoundaryCondition.SIMPLY_SUPPORTED,
            }

        self._boundary_conditions = boundary_conditions
        self.__h, self.__n = self._compute_grid_step()

    @property
    def h(self) -> float:
        """Get the grid step."""
        return self.__h

    @property
    def n(self) -> int:
        """Get the grid size."""
        return self.__n

    def build(self) -> Self:
        self._b, self._c = self.__build_update_matrices(self.__h, self.__n)

        return self

    def _compute_grid_step(self) -> tuple[float, int]:
        k = self._time_step
        gamma = self._gamma
        kappa = self._kappa
        _, b2 = self._loss

        a = (gamma * k) ** 2 + 4.0 * b2 * k
        h = sqrt(0.5 * (a + sqrt(a**2 + 16.0 * (kappa * k) ** 2)))
        n = int(1 / h)

        return 1 / n, n

    def __build_update_matrices(self, h: float, n: int) -> tuple[spmatrix, spmatrix]:
        k = self._time_step
        gamma = self._gamma
        kappa = self._kappa
        b1, b2 = self._loss

        lambda2 = (gamma * k / h) ** 2
        h2 = h**2
        mu2 = (kappa * k / h2) ** 2
        zeta = 2 * b2 * k / h2
        den = 1 + b1 * k

        alpha = 2 * b2 * (h / kappa) ** 2 / k
        beta = 2 + (gamma * h / kappa) ** 2 + alpha

        i = identity(n + 1)
        dxx = second_difference_matrix(n + 1)
        dxxxx = fourth_difference_matrix(n + 1)

        i, dxx, dxxxx = self.__apply_boundary_conditions((i, dxx, dxxxx), 2 * beta)

        b = self.__create_b_matrix((i, dxx, dxxxx), (lambda2, zeta, mu2)) / den
        c = self.__create_c_matrix((i, dxx), (b1, k, zeta, 2 * alpha, mu2), n) / -den

        return b, c

    def __create_b_matrix(
        self,
        matrices: tuple[spmatrix, spmatrix, spmatrix],
        coefficients: tuple[float, float, float],
    ) -> spmatrix:
        i, dxx, dxxxx = matrices
        lambda2, zeta, mu2 = coefficients

        return 2 * i + (lambda2 + zeta) * dxx - mu2 * dxxxx

    def __create_c_matrix(
        self,
        matrices: tuple[spmatrix, spmatrix],
        coefficients: tuple[float, ...],
        n: int,
    ) -> spmatrix:
        conditions = self._boundary_conditions
        i, dxx = matrices
        b1, k, zeta, alpha2, mu2 = coefficients

        c = (1.0 - b1 * k) * i + zeta * dxx

        if any(c == BoundaryCondition.FREE for c in conditions.values()):
            c = self.__add_free_matrix(c, n, (alpha2, mu2))

        return c

    def __add_free_matrix(self, c: spmatrix, n: int, coefficients: tuple[float, float]):
        conditions = self._boundary_conditions
        alpha2, mu2 = coefficients

        m = dok_matrix((n + 1, n + 1))

        if conditions[Endpoint.LEFT] == BoundaryCondition.FREE:
            m[0, 0] = -alpha2
            m[0, 1] = alpha2

        if conditions[Endpoint.RIGHT] == BoundaryCondition.FREE:
            m[-1, -1] = -alpha2
            m[-1, -2] = alpha2

        return c + mu2 * m.todia()

    def __apply_boundary_conditions(
        self,
        matrices: tuple[spmatrix, spmatrix, spmatrix],
        beta2: float,
    ) -> tuple[spmatrix, ...]:
        conditions = self._boundary_conditions

        i, dxx, dxxxx = [m.todok() for m in matrices]

        for endpoint in Endpoint:
            if conditions[endpoint] == BoundaryCondition.CLAMPED:
                self.__apply_clamped_condition((i, dxx, dxxxx), endpoint)

            if conditions[endpoint] == BoundaryCondition.SIMPLY_SUPPORTED:
                self.__apply_simply_supported_condition((i, dxx, dxxxx), endpoint)

            if conditions[endpoint] == BoundaryCondition.FREE:
                self.__apply_free_condition((i, dxx, dxxxx), endpoint, beta2)

        return tuple(m.todia() for m in (i, dxx, dxxxx))

    def __apply_clamped_condition(
        self, matrices: tuple[spmatrix, spmatrix, spmatrix], endpoint: Endpoint
    ) -> None:
        i, dxx, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            i[0, 0] = 0

            dxx[0, 0:2] = 0
            dxx[1, 0] = 0

            dxxxx[0, 0:3] = 0
            dxxxx[1, 0] = 0
            dxxxx[1, 1] = 7
            dxxxx[2, 0] = 0

        if endpoint == Endpoint.RIGHT:
            i[-1, -1] = 0

            dxx[-1, -2:] = 0
            dxx[-2, -1] = 0

            dxxxx[-1, -3:] = 0
            dxxxx[-2, -1] = 0
            dxxxx[-2, -2] = 7
            dxxxx[-3, -1] = 0

    def __apply_simply_supported_condition(
        self,
        matrices: tuple[spmatrix, spmatrix, spmatrix],
        endpoint: Endpoint,
    ) -> None:
        self.__apply_clamped_condition(matrices, endpoint)

        _, _, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            dxxxx[1, 1] = 5

        if endpoint == Endpoint.RIGHT:
            dxxxx[-2, -2] = 5

    def __apply_free_condition(
        self,
        matrices: tuple[spmatrix, spmatrix, spmatrix],
        endpoint: Endpoint,
        beta2: float,
    ) -> None:
        _, dxx, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            dxx[0, 0:2] = 0

            dxxxx[0, 0] = beta2 - 2
            dxxxx[0, 1] = -beta2
            dxxxx[0, 2] = 2
            dxxxx[1, 0] = -2
            dxxxx[1, 1] = 5

        if endpoint == Endpoint.RIGHT:
            dxx[-1, -2:] = 0

            dxxxx[-1, -1] = beta2 - 2
            dxxxx[-1, -2] = -beta2
            dxxxx[-1, -3] = 2
            dxxxx[-2, -1] = -2
            dxxxx[-2, -2] = 5


__all__ = ["BoundaryCondition", "Endpoint", "LinearResonator1D"]
