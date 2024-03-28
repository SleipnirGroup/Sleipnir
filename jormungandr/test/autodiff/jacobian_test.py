from jormungandr.autodiff import Jacobian, Variable, VariableMatrix
import jormungandr.autodiff as autodiff
import numpy as np
import pytest


def test_y_eq_x():
    y = VariableMatrix(3)
    x = VariableMatrix(3)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)

    # y = x
    #
    #         [1  0  0]
    # dy/dx = [0  1  0]
    #         [0  0  1]
    y = x
    J = Jacobian(y, x)

    expected_J = np.diag([1.0, 1.0, 1.0])
    assert (J.get().value() == expected_J).all()
    assert (J.value() == expected_J).all()


def test_y_eq_3x():
    y = VariableMatrix(3)
    x = VariableMatrix(3)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)

    # y = 3x
    #
    #         [3  0  0]
    # dy/dx = [0  3  0]
    #         [0  0  3]
    y = 3 * x
    J = Jacobian(y, x)

    expected_J = np.diag([3.0, 3.0, 3.0])
    assert (J.get().value() == expected_J).all()
    assert (J.value() == expected_J).all()


def test_products():
    y = VariableMatrix(3)
    x = VariableMatrix(3)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)

    #     [x₁x₂]
    # y = [x₂x₃]
    #     [x₁x₃]
    #
    #         [x₂  x₁  0 ]
    # dy/dx = [0   x₃  x₂]
    #         [x₃  0   x₁]
    #
    #         [2  1  0]
    # dy/dx = [0  3  2]
    #         [3  0  1]
    y[0] = x[0] * x[1]
    y[1] = x[1] * x[2]
    y[2] = x[0] * x[2]
    J = Jacobian(y, x)

    expected_J = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0], [3.0, 0.0, 1.0]])
    assert (J.get().value() == expected_J).all()
    assert (J.value() == expected_J).all()


@pytest.mark.skip(reason="Fails")
def test_nested_products():
    z = VariableMatrix(1)
    z[0].set_value(1)
    x = VariableMatrix(3)
    x[0] = 1 * z[0]
    x[1] = 2 * z[0]
    x[2] = 3 * z[0]

    J = Jacobian(x, z).value()
    assert J[0, 0] == 1.0
    assert J[1, 0] == 2.0
    assert J[2, 0] == 3.0

    #     [x₁x₂]
    # y = [x₂x₃]
    #     [x₁x₃]
    #
    #         [x₂  x₁  0 ]
    # dy/dx = [0   x₃  x₂]
    #         [x₃  0   x₁]
    #
    #         [2  1  0]
    # dy/dx = [0  3  2]
    #         [3  0  1]
    y = VariableMatrix(3)
    y[0] = x[0] * x[1]
    y[1] = x[1] * x[2]
    y[2] = x[0] * x[2]
    J = Jacobian(y, x)

    expected_J = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0], [3.0, 0.0, 1.0]])
    assert (J.get().value() == expected_J).all()
    assert (J.value() == expected_J).all()


def test_non_square():
    y = VariableMatrix(1)
    x = VariableMatrix(3)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)

    # y = [x₁ + 3x₂ − 5x₃]
    #
    # dy/dx = [1  3  −5]
    y[0] = x[0] + 3 * x[1] - 5 * x[2]
    J = Jacobian(y, x)

    expected_J = np.array([[1.0, 3.0, -5.0]])

    J_get_value = J.get().value()
    assert J_get_value.shape == (1, 3)
    assert (J_get_value == expected_J).all()

    J_value = J.get().value()
    assert J_value.shape == (1, 3)
    assert (J_value == expected_J).all()


def test_variable_reuse():
    y = VariableMatrix(1)
    x = VariableMatrix(2)

    # y = [x₁x₂]
    x[0].set_value(1)
    x[1].set_value(2)
    y[0] = x[0] * x[1]

    jacobian = Jacobian(y, x)

    # dy/dx = [x₂  x₁]
    # dy/dx = [2  1]
    J = jacobian.value()

    assert J.shape == (1, 2)
    assert J[0, 0] == 2.0
    assert J[0, 1] == 1.0

    x[0].set_value(2)
    x[1].set_value(1)
    # dy/dx = [x₂  x₁]
    # dy/dx = [1  2]
    J = jacobian.value()

    assert J.shape == (1, 2)
    assert J[0, 0] == 1.0
    assert J[0, 1] == 2.0
