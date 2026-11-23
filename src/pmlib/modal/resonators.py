"""Modal resonator definitions."""

from abc import ABC, abstractmethod
from math import ceil, floor

import numpy as np
import numpy.typing as npt

from ..resonators import Resonator


class ModalResonator(Resonator, ABC):
    """
    Abstract base class from which all concrete modal resonator classes inherit.
    """

    _omega_max = (2.0 / Resonator._k) * 0.99  # Eq. (37) largest eigenfrequency

    def __init__(
        self,
        gamma: float = 200,
        kappa: float = 1,
        sigma0: float = 1,
        sigma1: float = 1,
        M: float = 0.01,
        max_modes: int = 150,
    ):
        super().__init__(gamma, kappa, sigma0, sigma1, M)

        self.max_modes = max_modes

        self.omega, self.beta2, self.mu, self.damping_coefficients, self.num_modes = (
            self._compute_modes()
        )

        self._check_modes()

    @abstractmethod
    def _compute_modes(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
    ]:
        pass

    def _compute_omega(self, beta2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sqrt((self.gamma**2) * beta2 + (self.kappa**2) * beta2)  # Eq. (11)

    def _compute_damping_coefficients(
        self, beta2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return 2.0 * (self.sigma0 + self.sigma1 * beta2)

    """
    Sort modes in ascending order.
    """

    def _sort_modes(
        self,
        omega: npt.NDArray[np.float64],
        beta2: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        damping_coefficients: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], ...]:
        order = np.argsort(omega)

        omega = omega[order]
        beta2 = beta2[order]
        mu = mu[order]
        damping_coefficients = damping_coefficients[order]

        return omega, beta2, mu, damping_coefficients

    def _cap_modes(
        self,
        omega: npt.NDArray[np.float64],
        damping_coefficients: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool]:
        return (omega < self._omega_max) & (
            0.5 * ModalResonator._k * damping_coefficients < 0.999
        )

    def _check_modes(self) -> None:
        if self.num_modes == 0:
            raise ValueError(
                "No valid modes remain below the Nyquist/damping stability limits."
            )


class LinearResonator1D(ModalResonator):
    """
    A concrete implementation of a one dimensional linear resonator.

    Parameters
    ----------
    gamma : float, default 200.0
        The spatially scaled wave speed c / L (s^-1).
    kappa : float, default 1.0
        The spatially scaled stiffness coefficient (s^-1).
    sigma0 : float, default 1.0
        The frequency independent damping constant (s^-1).
    sigma1 : float, default 1.0
        The frequency dependent damping constant (s^-1).
    M : float, default 0.01
        The total mass (kg) of the resonator (for physical impedance scaling).
    max_modes : int, default 150
        The maximum number of modes to retain.
    """

    DIMENSION = 1

    def _compute_modes(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
    ]:
        mu = np.arange(1, self.max_modes + 1, dtype=np.float64)
        beta2 = (mu * np.pi) ** 2
        mu = mu[:, None]

        omega, beta2, mu, damping_coefficients = self._sort_modes(
            self._compute_omega(beta2),
            beta2,
            mu,
            self._compute_damping_coefficients(beta2),
        )

        valid = self._cap_modes(omega, damping_coefficients)

        omega = omega[valid]

        return (
            omega,
            beta2[valid],
            mu[valid],
            damping_coefficients[valid],
            len(omega),
        )


class LinearResonator2D(ModalResonator):
    """
    A concrete implementation of a two dimensional linear resonator.

    Parameters
    ----------
    gamma : float, default 200.0
        The spatially scaled wave speed c / sqrt(Area) (s^-1).
    kappa : float, default 1.0
        The spatially scaled stiffness coefficient (s^-1).
    sigma0 : float, default 1.0
        The frequency independent damping constant (s^-1).
    sigma1 : float, default 1.0
        The frequency dependent damping constant (s^-1).
    M : float, default 0.01
        The total mass (kg) of the resonator (for physical impedance scaling).
    epsilon : float, default 1.0
        The aspect ratio L_x / L_y
    max_modes : int, default 150
        The maximum number of modes to retain.
    """

    DIMENSION = 2

    def __init__(
        self,
        gamma: float = 200,
        kappa: float = 1,
        sigma0: float = 1,
        sigma1: float = 1,
        M: float = 0.01,
        epsilon: float = 1.0,
        max_modes: int = 150,
    ):
        super().__init__(gamma, kappa, sigma0, sigma1, M, max_modes)

        if epsilon <= 0:
            raise ValueError("Aspect ratio epsilon must be greater than zero.")

        self.epsilon = epsilon

    def __compute_beta2_max(self) -> float:
        if not self.kappa > 0.0:
            return (self._omega_max / self.gamma) ** 2

        return (
            -(self.gamma**2)
            + np.sqrt((self.gamma**4) + 4.0 * (self.kappa**2) * (self._omega_max**2))
        ) / (2.0 * (self.kappa**2))

    def __compute_m_bounds(self, beta2m: float) -> tuple[int, int]:
        """
        Compute tight aspect ratio aware bounds from both frequency cutoff
        bounding rectangle and boundary padded quarter ellipse mode count
        bounding rectangle.
        """

        # Frequency cutoff bounding rectangle
        m1f = max(1, floor(np.sqrt(beta2m * self.epsilon) / np.pi))
        m2f = max(1, floor(np.sqrt(beta2m / self.epsilon) / np.pi))

        # Boundary padded quarter ellipse mode count bounding rectangle
        target_area = 2.0 * self.max_modes

        m1c = ceil(np.sqrt((4.0 / np.pi) * target_area * self.epsilon)) + 4
        m2c = ceil(np.sqrt((4.0 / np.pi) * target_area / self.epsilon)) + 4

        return max(1, min(m1f, m1c)), max(1, min(m2f, m2c))

    def _compute_modes(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
    ]:
        beta2m = self.__compute_beta2_max()

        m1m, m2m = self.__compute_m_bounds(beta2m)

        # Generate a (m1m * m2m) X 2 matrix of the form [[1, 1], [1, 2], ... [1, m2m], ..., [m1m, 1], [m1m, 2], ..., [m1m, m2m]]
        mu = (np.indices((m1m, m2m), dtype=np.float64) + 1).reshape(2, -1).T

        beta2 = (np.pi**2) * (
            (mu[:, 0] ** 2) / self.epsilon + self.epsilon * (mu[:, 1] ** 2)
        )

        omega, beta2, mu, damping_coefficients = self._sort_modes(
            self._compute_omega(beta2),
            beta2,
            mu,
            self._compute_damping_coefficients(beta2),
        )

        valid = self._cap_modes(omega, damping_coefficients)

        omega = omega[valid][: self.max_modes]

        return (
            omega,
            beta2[valid][: self.max_modes],
            mu[valid][: self.max_modes],
            damping_coefficients[valid][: self.max_modes],
            len(self.omega),
        )
