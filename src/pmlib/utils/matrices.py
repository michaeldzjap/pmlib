"""Sparse difference matrix utilities."""

from numpy import full, ones, prod, vstack
from scipy.sparse import dia_matrix, spmatrix

import numpy as np


def second_difference_matrix(n: int = 2) -> dia_matrix[np.floating]:
    """
    Generate the second order difference operator in matrix form.

    Parameters
    ----------
    n : int, default 2
        The number of rows and columns of the matrix.

    Returns
    -------
    spmatrix
        The second difference matrix.

    Raises
    ------
    ValueError
        If `n` is smaller than 2.
    """
    if n < 2:
        raise ValueError("n must be equal or greater than 2")

    diagonals = vstack([ones(n), full(n, -2.0), ones(n)])

    return dia_matrix((diagonals, (-1.0, 0.0, 1.0)), shape=(n, n))


def fourth_difference_matrix(n: int = 3) -> dia_matrix[np.floating]:
    """
    Generate the fourth order difference operator in matrix form.

    Parameters
    ----------
    n : int, default 3
        The number of rows and columns of the matrix.

    Returns
    -------
    spmatrix
        The fourth difference matrix.

    Raises
    ------
    ValueError
        If `n` is smaller than 3.
    """
    if n < 3:
        raise ValueError("n must be equal or greater than 3")

    diagonals = vstack([ones(n), full(n, -4.0), full(n, 6.0), full(n, -4.0), ones(n)])

    return dia_matrix((diagonals, (-2, -1, 0, 1, 2)), shape=(n, n))


def laplacian_matrix(n: tuple[int, int] = (2, 2)) -> dia_matrix[np.floating]:
    """
    Generate the discrete laplacian operator in matrix form.

    Parameters
    ----------
    n : tuple, default (2, 2)
        A tuple whose product defines the number of rows and columns of the matrix.

    Returns
    -------
    spmatrix
        The laplacian matrix.

    Raises
    ------
    ValueError
        If `n[i]` is smaller than 2.
    """
    for i, ni in enumerate(n):
        if ni < 2:
            raise ValueError(f"n[{i}] must be equal or greater than 2")

    m = prod(n)

    diagonals = np.vstack(
        [
            np.ones(m),
            np.ones(m),
            np.full(m, -4.0),
            np.ones(m),
            np.ones(m),
        ]
    )

    return dia_matrix((diagonals, (-n[1], -1, 0, 1, n[1])), shape=(m, m))


def biharmonic_matrix(n: tuple[int, int] = (4, 4)) -> spmatrix:
    """
    Generate the discrete biharmonic operator in matrix form.

    Parameters
    ----------
    n : tuple, default (4, 4)
        A tuple whose product defines the number of rows and columns of the matrix.

    Returns
    -------
    spmatrix
        The biharmonic matrix.

    Raises
    ------
    ValueError
        If `n[i]` is smaller than 4.
    """
    for i, ni in enumerate(n):
        if ni < 4:
            raise ValueError(f"n[{i}] must be equal or greater than 4")

    m = prod(n)

    diagonals = np.vstack(
        [
            np.ones(m),
            np.full(m, 2.0),
            np.full(m, -8.0),
            np.full(m, 2.0),
            np.ones(m),
            np.full(m, -8.0),
            np.full(m, 20.0),
            np.full(m, -8.0),
            np.ones(m),
            np.full(m, 2.0),
            np.full(m, -8.0),
            np.full(m, 2.0),
            np.ones(m),
        ]
    )

    return dia_matrix(
        (
            diagonals,
            (
                2 * -n[1],
                -n[1] - 1,
                -n[1],
                -n[1] + 1,
                -2,
                -1,
                0,
                1,
                2,
                n[1] - 1,
                n[1],
                n[1] + 1,
                2 * n[1],
            ),
        ),
        shape=(m, m),
    )


__all__ = [
    "second_difference_matrix",
    "fourth_difference_matrix",
    "laplacian_matrix",
    "biharmonic_matrix",
]
