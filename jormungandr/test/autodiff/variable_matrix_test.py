import numpy as np

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import VariableMatrix


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

    # Slice from start
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

    # Slice from start with step of 2
    s = mat[:, ::2]
    assert s.shape == (4, 2)
    assert (
        s.value() == np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]])
    ).all()

    # Slice from end with negative step for row and column
    s = mat[::-1, ::-2]
    assert s.shape == (4, 2)
    assert (
        s.value()
        == np.array(
            [
                [16.0, 14.0],
                [12.0, 10.0],
                [8.0, 6.0],
                [4.0, 2.0],
            ]
        )
    ).all()

    # Slice from start and column -1
    s = mat[1:, -1]
    assert s.shape == (3, 1)
    assert (s.value() == np.array([[8.0], [12.0], [16.0]])).all()

    # Slice from start and column -2
    s = mat[1:, -2]
    assert s.shape == (3, 1)
    assert (s.value() == np.array([[7.0], [11.0], [15.0]])).all()

    # Block assignment
    assert mat[::2, ::2].shape == (2, 2)
    mat[::2, ::2] = np.array([[17.0, 18.0], [19.0, 20.0]])
    assert (
        mat.value()
        == np.array(
            [
                [17.0, 2.0, 18.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [19.0, 10.0, 20.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
    ).all()


def test_subslicing():
    # Block-of-block assignment (row skip forward)
    mat = VariableMatrix(5, 5)
    assert mat[::2, ::1][1:3, 1:4].shape == (2, 3)
    mat[::2, ::1][1:3, 1:4] = np.array([[1, 2, 3], [4, 5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 5, 6, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (row skip backward)
    mat = VariableMatrix(5, 5)
    assert mat[::-2, ::-1][1:3, 1:4].shape == (2, 3)
    mat[::-2, ::-1][1:3, 1:4] = np.array([[1, 2, 3], [4, 5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 6, 5, 4, 0],
                [0, 0, 0, 0, 0],
                [0, 3, 2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (column skip forward)
    mat = VariableMatrix(5, 5)
    assert mat[::1, ::2][1:4, 1:3].shape == (3, 2)
    mat[::1, ::2][1:4, 1:3] = np.array([[1, 2], [3, 4], [5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 2],
                [0, 0, 3, 0, 4],
                [0, 0, 5, 0, 6],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (column skip backward)
    mat = VariableMatrix(5, 5)
    assert mat[::-1, ::-2][1:4, 1:3].shape == (3, 2)
    mat[::-1, ::-2][1:4, 1:3] = np.array([[1, 2], [3, 4], [5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [6, 0, 5, 0, 0],
                [4, 0, 3, 0, 0],
                [2, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()


def test_iterators():
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    sub_A = A[2:3, 1:3]

    assert sum(1 for _ in A) == 9
    assert sum(1 for _ in sub_A) == 2

    i = 1
    for elem in A:
        assert elem.value() == i
        i += 1

    i = 8
    for elem in sub_A:
        assert elem.value() == i
        i += 1


def test_value():
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Full matrix
    assert (A.value() == expected).all()
    assert A.value(3) == 4.0
    assert A.T.value(3) == 2.0

    # Slice
    assert (A[1:3, 1:3].value() == expected[1:3, 1:3]).all()
    assert A[1:3, 1:3].value(2) == 8.0
    assert A[1:3, 1:3].T.value(2) == 6.0

    # Slice-of-slice
    assert (A[1:3, 1:3][:, 1:].value() == expected[1:3, 1:3][:, 1:]).all()
    assert A[1:3, 1:3][:, 1:].value(1) == 9.0
    assert A[1:3, 1:3].T[:, 1:].value(1) == 9.0


def test_cwise_map():
    # VariableMatrix CwiseTransform
    A = VariableMatrix([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]])

    result1 = A.cwise_map(autodiff.abs)
    expected1 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()

    assert (result1.value() == expected1).all()

    # VariableBlock CwiseTransform
    sub_A = A[:2, :2]

    result2 = sub_A.cwise_map(autodiff.abs)
    expected2 = np.array([[2.0, 3.0], [5.0, 6.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()
    assert (sub_A.value() == -expected2).all()

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


def check_solve(A: VariableMatrix, B: VariableMatrix):
    print(f"Solve {A.shape[0]}x{A.shape[1]}")

    X = autodiff.solve(A, B)

    assert X.shape == (A.shape[1], B.shape[1])
    assert np.linalg.norm(A.value() @ X.value() - B.value()) < 1e-12


def test_solve_free_function():
    # 1x1 special case
    check_solve(VariableMatrix([[2.0]]), VariableMatrix([[5.0]]))

    # 2x2 special case
    check_solve(
        VariableMatrix([[1.0, 2.0], [3.0, 4.0]]), VariableMatrix([[5.0], [6.0]])
    )

    # 3x3 special case
    check_solve(
        VariableMatrix([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0], [7.0, 8.0, 9.0]]),
        VariableMatrix([[10.0], [11.0], [12.0]]),
    )

    # 4x4 special case
    check_solve(
        VariableMatrix(
            [
                [1.0, 2.0, 3.0, -4.0],
                [-5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        ),
        VariableMatrix([[17.0], [18.0], [19.0], [20.0]]),
    )

    # 5x5 general case
    check_solve(
        VariableMatrix(
            [
                [1.0, 2.0, 3.0, -4.0, 5.0],
                [-5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0],
            ]
        ),
        VariableMatrix([[21.0], [22.0], [23.0], [24.0], [25.0]]),
    )
