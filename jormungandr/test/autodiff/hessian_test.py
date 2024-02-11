from functools import reduce
import math
import operator

from jormungandr.autodiff import Gradient, Hessian, Variable, VariableMatrix
import jormungandr.autodiff as autodiff
import numpy as np
import pytest


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def test_linear():
    # y = x
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0]

    # dy/dx = 1
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 1.0

    # d²y/dx² = 0
    H = Hessian(y, x).value()
    assert H[0, 0] == 0.0


def test_quartic():
    # y = x²
    # y = x * x
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0]

    # dy/dx = x (rhs) + x (lhs)
    #       = (3) + (3)
    #       = 6
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 6.0

    # d²y/dx² = d/dx(x (rhs) + x (lhs))
    #         = 1 + 1
    #         = 2
    H = Hessian(y, x).value()
    assert H[0, 0] == 2.0


def test_sum():
    y = Variable()
    x = VariableMatrix(5)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)
    x[3].set_value(4)
    x[4].set_value(5)

    y = sum(x)
    g = Gradient(y, x).value()

    assert y.value() == 15.0
    for i in range(x.rows()):
        assert g[i] == 1.0

    H = Hessian(y, x).value()
    for i in range(x.rows()):
        for j in range(x.rows()):
            assert H[i, j] == 0.0


def test_sum_of_products():
    y = Variable()
    x = VariableMatrix(5)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)
    x[3].set_value(4)
    x[4].set_value(5)

    # y = ||x||²
    y = (x.T @ x)[0, 0]
    g = Gradient(y, x).value()

    assert y.value() == 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5
    for i in range(x.rows()):
        assert g[i] == (2 * x[i]).value()

    H = Hessian(y, x).value()
    for i in range(x.rows()):
        for j in range(x.rows()):
            if i == j:
                assert H[i, j] == 2.0
            else:
                assert H[i, j] == 0.0


def test_product_of_sines():
    y = Variable()
    x = VariableMatrix(5)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)
    x[3].set_value(4)
    x[4].set_value(5)

    # y = prod(sin(x))
    y = prod(x.cwise_transform(autodiff.sin))
    g = Gradient(y, x).value()

    assert y.value() == math.sin(1) * math.sin(2) * math.sin(3) * math.sin(
        4
    ) * math.sin(5)
    for i in range(x.rows()):
        assert g[i, 0] == pytest.approx((y / autodiff.tan(x[i])).value(), abs=1e-15)

    H = Hessian(y, x).value()
    for i in range(x.rows()):
        for j in range(x.rows()):
            if i == j:
                assert H[i, j] == pytest.approx(
                    (g[i, 0] / autodiff.tan(x[i])).value()
                    * (1.0 - 1.0 / (autodiff.cos(x[i]) * autodiff.cos(x[i]))).value(),
                    abs=1e-15,
                )
            else:
                assert H[i, j] == pytest.approx(
                    (g[j, 0] / autodiff.tan(x[i])).value(), abs=1e-15
                )


def test_sum_of_squared_residuals():
    y = Variable()
    x = VariableMatrix(5)
    x[0].set_value(1)
    x[1].set_value(1)
    x[2].set_value(1)
    x[3].set_value(1)
    x[4].set_value(1)

    # y = sum(diff(x).^2)
    y = sum((x[:4, :1] - x[1:5, :1]).cwise_transform(lambda x: x**2))
    g = Gradient(y, x).value()

    assert y.value() == 0.0
    assert g[0, 0] == (2 * x[0] - 2 * x[1]).value()
    assert g[1, 0] == (-2 * x[0] + 4 * x[1] - 2 * x[2]).value()
    assert g[2, 0] == (-2 * x[1] + 4 * x[2] - 2 * x[3]).value()
    assert g[3, 0] == (-2 * x[2] + 4 * x[3] - 2 * x[4]).value()
    assert g[4, 0] == (-2 * x[3] + 2 * x[4]).value()

    H = Hessian(y, x).value()

    expected_H = np.array(
        [
            [2.0, -2.0, 0.0, 0.0, 0.0],
            [-2.0, 4.0, -2.0, 0.0, 0.0],
            [0.0, -2.0, 4.0, -2.0, 0.0],
            [0.0, 0.0, -2.0, 4.0, -2.0],
            [0.0, 0.0, 0.0, -2.0, 2.0],
        ]
    )
    for i in range(x.rows()):
        for j in range(x.rows()):
            assert H[i, j] == expected_H[i, j]


def test_sum_of_squares():
    r = VariableMatrix(4)
    r[0].set_value(25.0)
    r[1].set_value(10.0)
    r[2].set_value(5.0)
    r[3].set_value(0.0)
    x = VariableMatrix(4)
    x[0].set_value(0.0)
    x[1].set_value(0.0)
    x[2].set_value(0.0)
    x[3].set_value(0.0)

    J = 0.0
    for i in range(4):
        J += (r[i] - x[i]) * (r[i] - x[i])

    H = Hessian(J, x).value()
    for row in range(4):
        for col in range(4):
            if row == col:
                assert H[row, col] == 2.0
            else:
                assert H[row, col] == 0.0


def test_rosenbrock():
    input = VariableMatrix(2)
    x = input[0]
    y = input[1]

    for x0 in np.arange(-2.5, 2.5, 0.1):
        for y0 in np.arange(-2.5, 2.5, 0.1):
            x.set_value(x0)
            y.set_value(y0)
            z = (1 - x) ** 2 + 100 * (y - x**2) ** 2

            H = Hessian(z, input).value()
            assert H[0, 0] == pytest.approx(
                -400 * (y0 - x0 * x0) + 800 * x0 * x0 + 2, abs=1e-12
            )
            assert H[0, 1] == -400 * x0
            assert H[1, 0] == -400 * x0
            assert H[1, 1] == 200


def test_reuse():
    y = Variable()
    x = VariableMatrix(1)

    # y = x³
    x[0].set_value(1)
    y = x[0] * x[0] * x[0]

    hessian = Hessian(y, x)

    # d²y/dx² = 6x
    # H = 6
    H = hessian.value()

    assert H.shape == (1, 1)
    assert H[0, 0] == 6.0

    x[0].set_value(2)
    # d²y/dx² = 6x
    # H = 12
    H = hessian.value()

    assert H.shape == (1, 1)
    assert H[0, 0] == 12.0
