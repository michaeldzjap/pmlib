"""Sparse difference matrix utilities."""

from numpy import prod
from scipy.sparse import diags, spmatrix


def second_difference_matrix(n: int = 2) -> spmatrix:
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

    return diags((1, -2, 1), (-1, 0, 1), (n, n))


def fourth_difference_matrix(n: int = 3) -> spmatrix:
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

    return diags((1, -4, 6, -4, 1), (-2, -1, 0, 1, 2), (n, n))


def laplacian_matrix(n: tuple[int, int] = (2, 2)) -> spmatrix:
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

    return diags((1, 1, -4, 1, 1), (-n[1], -1, 0, 1, n[1]), (m, m))


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

    return diags(
        (1, 2, -8, 2, 1, -8, 20, -8, 1, 2, -8, 2, 1),
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
        (m, m),
    )


__all__ = [
    "second_difference_matrix",
    "fourth_difference_matrix",
    "laplacian_matrix",
    "biharmonic_matrix",
]
