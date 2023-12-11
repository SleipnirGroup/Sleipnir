from jormungandr.autodiff import VariableMatrix
import jormungandr.autodiff as autodiff

import numpy as np


def test_assignment_to_default():
    mat = VariableMatrix()

    assert mat.rows() == 0
    assert mat.cols() == 0

    mat = VariableMatrix(2, 2)

    assert mat.rows() == 2
    assert mat.cols() == 2
    assert mat[0, 0] == 0.0
    assert mat[0, 1] == 0.0
    assert mat[1, 0] == 0.0
    assert mat[1, 1] == 0.0

    mat[0, 0].set_value(1.0)
    mat[0, 1].set_value(2.0)
    mat[1, 0].set_value(3.0)
    mat[1, 1].set_value(4.0)

    assert mat[0, 0] == 1.0
    assert mat[0, 1] == 2.0
    assert mat[1, 0] == 3.0
    assert mat[1, 1] == 4.0


def test_cwise_transform():
    # VariableMatrix CwiseTransform
    A = VariableMatrix(2, 3)
    A.set_value(np.array([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]]))

    result1 = A.cwise_transform(autodiff.abs)
    expected1 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Don't modify original matrix
    assert (-expected1 == A.value()).all()

    assert (expected1 == result1.value()).all()

    # VariableBlock CwiseTransform
    Asub = A[:2, :2]

    result2 = Asub.cwise_transform(autodiff.abs)
    expected2 = np.array([[2.0, 3.0], [5.0, 6.0]])

    # Don't modify original matrix
    assert (-expected1 == A.value()).all()
    assert (-expected2 == Asub.value()).all()

    assert (expected2 == result2.value()).all()


def test_cwise_reduce():
    A = VariableMatrix(2, 3)
    A.set_value(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    B = VariableMatrix(2, 3)
    B.set_value(np.array([[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]))
    result = autodiff.cwise_reduce(A, B, lambda a, b: a * b)

    assert (np.array([[16.0, 27.0, 40.0], [55.0, 72.0, 91.0]]) == result.value()).all()
