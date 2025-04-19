"""Test sparse difference matrix utilities."""

from numpy import array
from numpy.testing import assert_array_equal

from pmlib.utils.matrices import (
    biharmonic_matrix,
    fourth_difference_matrix,
    laplacian_matrix,
    second_difference_matrix,
)


def test_second_difference_matrix_coefficients() -> None:
    """
    Assert that the second difference matrix contains the expected
    coefficients at all relevant indices.
    """
    matrix = second_difference_matrix()

    assert_array_equal(array([-2, -2]), matrix.diagonal(0))
    assert_array_equal(array([1]), matrix.diagonal(1))
    assert_array_equal(array([1]), matrix.diagonal(-1))


def test_second_difference_matrix_shape() -> None:
    """
    Assert that the second difference matrix has the expected shape.
    """
    matrix = second_difference_matrix(3)

    assert (3, 3) == matrix.shape


def test_fourth_difference_matrix_coefficients() -> None:
    """
    Assert that the fourth difference matrix contains the expected coefficients
    at all relevant indices.
    """
    matrix = fourth_difference_matrix()

    assert_array_equal(array([6, 6, 6]), matrix.diagonal(0))
    assert_array_equal(array([-4, -4]), matrix.diagonal(1))
    assert_array_equal(array([-4, -4]), matrix.diagonal(-1))
    assert_array_equal(array([1]), matrix.diagonal(2))
    assert_array_equal(array([1]), matrix.diagonal(-2))


def test_fourth_difference_matrix_shape() -> None:
    """
    Assert that the fourth difference matrix has the expected shape.
    """
    matrix = fourth_difference_matrix(4)

    assert (4, 4) == matrix.shape


def test_laplacian_matrix_coefficients() -> None:
    """
    Assert that the Laplacian matrix contains the expected coefficients at all
    relevant indices.
    """
    matrix = laplacian_matrix()

    assert_array_equal(array(4 * [-4]), matrix.diagonal(0))
    assert_array_equal(array(3 * [1]), matrix.diagonal(1))
    assert_array_equal(array(3 * [1]), matrix.diagonal(-1))
    assert_array_equal(array(2 * [1]), matrix.diagonal(2))
    assert_array_equal(array(2 * [1]), matrix.diagonal(-2))


def test_laplacian_matrix_shape() -> None:
    """
    Assert that the Laplacian matrix has the expected shape.
    """
    matrix = laplacian_matrix((3, 3))

    assert (9, 9) == matrix.shape


def test_biharmonic_matrix_coefficients() -> None:
    """
    Assert that the biharmonic matrix contains the expected coefficients at all
    relevant indices.
    """
    matrix = biharmonic_matrix()

    assert_array_equal(array(16 * [20]), matrix.diagonal(0))
    assert_array_equal(array(15 * [-8]), matrix.diagonal(1))
    assert_array_equal(array(15 * [-8]), matrix.diagonal(-1))
    assert_array_equal(array(14 * [1]), matrix.diagonal(2))
    assert_array_equal(array(14 * [1]), matrix.diagonal(-2))
    assert_array_equal(array(13 * [2]), matrix.diagonal(3))
    assert_array_equal(array(13 * [2]), matrix.diagonal(-3))
    assert_array_equal(array(12 * [-8]), matrix.diagonal(4))
    assert_array_equal(array(12 * [-8]), matrix.diagonal(-4))
    assert_array_equal(array(11 * [2]), matrix.diagonal(5))
    assert_array_equal(array(11 * [2]), matrix.diagonal(-5))
    assert_array_equal(array(8 * [1]), matrix.diagonal(8))
    assert_array_equal(array(8 * [1]), matrix.diagonal(-8))


def test_biharmonic_matrix_shape() -> None:
    """
    Assert that the biharmonic matrix has the expected shape.
    """
    matrix = biharmonic_matrix((5, 5))

    assert (25, 25) == matrix.shape
