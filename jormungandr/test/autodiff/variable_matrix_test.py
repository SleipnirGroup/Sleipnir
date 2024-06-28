from jormungandr.autodiff import VariableMatrix
import jormungandr.autodiff as autodiff

import numpy as np


def test_assignment_to_default():
    mat = VariableMatrix()

    assert mat.rows() == 0
    assert mat.cols() == 0
    assert mat.shape == (0, 0)

    mat = VariableMatrix(2, 2)

    assert mat.rows() == 2
    assert mat.cols() == 2
    assert mat.shape == (2, 2)
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


def test_slicing():
    mat = VariableMatrix(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    assert mat.shape == (4, 4)

    # Single-arg index operator on full matrix
    for i in range(mat.shape[0] * mat.shape[1]):
        assert mat[i] == i + 1

    # Slice from beginning
    s = mat[1:, 2:]
    assert s.shape == (3, 2)
    # Single-arg index operator on forward slice
    assert s[0] == 7.0
    assert s[1] == 8.0
    assert s[2] == 11.0
    assert s[3] == 12.0
    assert s[4] == 15.0
    assert s[5] == 16.0
    # Double-arg index operator on forward slice
    assert s[0, 0] == 7.0
    assert s[0, 1] == 8.0
    assert s[1, 0] == 11.0
    assert s[1, 1] == 12.0
    assert s[2, 0] == 15.0
    assert s[2, 1] == 16.0

    # Slice from end
    s = mat[-1:, -2:]
    assert s.shape == (1, 2)
    # Single-arg index operator on reverse slice
    assert s[0] == 15.0
    assert s[1] == 16.0
    # Double-arg index operator on reverse slice
    assert s[0, 0] == 15.0
    assert s[0, 1] == 16.0


def test_subslicing():
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Block assignment
    assert A[1:3, 1:3].shape == (2, 2)
    A[1:3, 1:3] = np.array([[10.0, 11.0], [12.0, 13.0]])

    expected1 = np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 11.0], [7.0, 12.0, 13.0]])
    assert (expected1 == A.value()).all()

    # Block-of-block assignment
    assert A[1:3, 1:3][1:, 1:].shape == (1, 1)
    A[1:3, 1:3][1:, 1:] = 14.0

    expected2 = np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 11.0], [7.0, 12.0, 14.0]])
    assert (A.value() == expected2).all()


def test_iterators():
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # VariableMatrix iterator
    assert sum(1 for e in A) == 9

    i = 1
    for elem in A:
        assert elem.value() == i
        i += 1

    Asub = A[2:3, 1:3]

    # VariableBlock iterator
    assert sum(1 for e in Asub) == 2

    i = 8
    for elem in Asub:
        assert elem.value() == i
        i += 1


def test_cwise_transform():
    # VariableMatrix CwiseTransform
    A = VariableMatrix([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]])

    result1 = A.cwise_transform(autodiff.abs)
    expected1 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()

    assert (result1.value() == expected1).all()

    # VariableBlock CwiseTransform
    Asub = A[:2, :2]

    result2 = Asub.cwise_transform(autodiff.abs)
    expected2 = np.array([[2.0, 3.0], [5.0, 6.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()
    assert (Asub.value() == -expected2).all()

    assert (result2.value() == expected2).all()


def test_zero_static_function():
    A = VariableMatrix.zero(2, 3)

    for row in range(A.rows()):
        for col in range(A.cols()):
            assert A[row, col].value() == 0.0


def test_ones_static_function():
    A = VariableMatrix.ones(2, 3)

    for row in range(A.rows()):
        for col in range(A.cols()):
            assert A[row, col].value() == 1.0


def test_cwise_reduce():
    A = VariableMatrix([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    B = VariableMatrix([[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
    result = autodiff.cwise_reduce(A, B, lambda a, b: a * b)

    assert (result.value() == np.array([[16.0, 27.0, 40.0], [55.0, 72.0, 91.0]])).all()


def test_block_free_function():
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = VariableMatrix([[7.0], [8.0]])

    mat1 = autodiff.block([[A, B]])
    expected1 = np.array([[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0]])
    assert mat1.shape == (2, 4)
    assert (mat1.value() == expected1).all()

    C = VariableMatrix([[9.0, 10.0, 11.0, 12.0]])

    mat2 = autodiff.block([[A, B], [C]])
    expected2 = np.array(
        [[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    assert mat2.shape == (3, 4)
    assert (mat2.value() == expected2).all()


def test_solve_free_function():
    A_11 = VariableMatrix([[2.0]])
    B_11 = VariableMatrix([[5.0]])
    X_11 = autodiff.solve(A_11, B_11)

    expected_11 = np.array([[2.5]])
    assert X_11.shape == (1, 1)
    assert (A_11.value() @ X_11.value() == B_11.value()).all()
    assert (X_11.value() == expected_11).all()

    A_22 = VariableMatrix([[1.0, 2.0], [3.0, 4.0]])
    B_21 = VariableMatrix([[5.0], [6.0]])
    X_21 = autodiff.solve(A_22, B_21)

    expected_21 = np.array([[-4.0], [4.5]])
    assert X_21.shape == (2, 1)
    assert (A_22.value() @ X_21.value() == B_21.value()).all()
    assert (X_21.value() == expected_21).all()

    A_33 = VariableMatrix([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0], [7.0, 8.0, 9.0]])
    B_31 = VariableMatrix([[10.0], [11.0], [12.0]])
    X_31 = autodiff.solve(A_33, B_31)

    expected_31 = np.array([[-7.5], [6.0], [11.0 / 6.0]])
    assert X_31.shape == (3, 1)
    assert np.linalg.norm(A_33.value() @ X_31.value() - B_31.value()) < 1e-12
    assert np.linalg.norm(X_31.value() - expected_31) < 1e-12

    A_44 = VariableMatrix(
        [
            [1.0, 2.0, 3.0, -4.0],
            [-5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    B_41 = VariableMatrix([[17.0], [18.0], [19.0], [20.0]])
    X_41 = autodiff.solve(A_44, B_41)

    expected_41 = np.array([[4.44089e-16], [-16.25], [16.5], [0.0]])
    assert X_41.shape == (4, 1)
    assert np.linalg.norm(A_44.value() @ X_41.value() - B_41.value()) < 1e-12
    assert np.linalg.norm(X_41.value() - expected_41) < 1e-12
