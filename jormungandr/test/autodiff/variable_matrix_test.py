from jormungandr.autodiff import VariableMatrix


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

    mat[0, 0].set(1.0)
    mat[0, 1].set(2.0)
    mat[1, 0].set(3.0)
    mat[1, 1].set(4.0)

    assert mat[0, 0] == 1.0
    assert mat[0, 1] == 2.0
    assert mat[1, 0] == 3.0
    assert mat[1, 1] == 4.0
