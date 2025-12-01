import math
import operator
from functools import reduce

import numpy as np
import pytest
import sleipnir.autodiff as autodiff
from sleipnir.autodiff import Gradient, Hessian, Jacobian, Variable, VariableMatrix


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
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 0.0
    assert H.value()[0, 0] == 0.0


def test_quadratic():
    # y = x²
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0]

    # dy/dx = 2x = 6
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 6.0

    # d²y/dx² = 2
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 2.0
    assert H.value()[0, 0] == 2.0


def test_cubic():
    # y = x³
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0] * x[0]

    # dy/dx = 3x² = 27
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 27.0

    # d²y/dx² = 6x = 18
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 18.0
    assert H.value()[0, 0] == 18.0


def test_quartic():
    # y = x⁴
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0] * x[0] * x[0]

    # dy/dx = 4x³ = 108
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 108.0

    # d²y/dx² = 12x² = 108
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 108.0
    assert H.value()[0, 0] == 108.0


def test_sum():
    y = Variable()
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    y = sum(x)
    assert y.value() == 15.0

    g = Gradient(y, x)
    assert (g.get().value() == np.full((5, 1), 1.0)).all()
    assert (g.value() == np.full((5, 1), 1.0)).all()

    H = Hessian(y, x)
    assert (H.get().value() == np.zeros((5, 5))).all()
    assert (H.value() == np.zeros((5, 5))).all()


def test_sum_of_products():
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    # y = ||x||²
    y = (x.T @ x)[0, 0]
    assert y.value() == sum([x * x for x in range(1, 6)])

    g = Gradient(y, x)
    assert (g.get().value() == 2 * x.value()).all()
    assert (g.value() == 2 * x.value()).all()

    H = Hessian(y, x)

    expected_H = np.diag([2.0] * 5)
    assert (H.get().value() == expected_H).all()
    assert (H.value() == expected_H).all()


def test_product_of_sines():
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    # y = prod(sin(x))
    y = prod(x.cwise_map(autodiff.sin))
    assert y.value() == pytest.approx(prod(math.sin(x) for x in range(1, 6)), abs=1e-15)

    g = Gradient(y, x)
    for i in range(x.rows()):
        assert g.get().value()[i, 0] == pytest.approx(
            y.value() / math.tan(x[i].value()), abs=1e-15
        )
        assert g.value()[i, 0] == pytest.approx(
            y.value() / math.tan(x[i].value()), abs=1e-15
        )

    H = Hessian(y, x)

    expected_H = np.empty((5, 5))
    for i in range(x.rows()):
        for j in range(x.rows()):
            if i == j:
                expected_H[i, j] = -y.value()
            else:
                expected_H[i, j] = y.value() / (
                    math.tan(x[i].value()) * math.tan(x[j].value())
                )
    np.testing.assert_allclose(H.get().value(), expected_H)
    np.testing.assert_allclose(H.value().todense(), expected_H)


def test_sum_of_squared_residuals():
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(1)

    # y = sum(diff(x).^2)
    y = sum((x[:4, :1] - x[1:5, :1]).cwise_map(lambda x: x**2))
    g = Gradient(y, x).value()

    assert y.value() == 0.0
    assert g[0, 0] == 2 * x[0].value() - 2 * x[1].value()
    assert g[1, 0] == -2 * x[0].value() + 4 * x[1].value() - 2 * x[2].value()
    assert g[2, 0] == -2 * x[1].value() + 4 * x[2].value() - 2 * x[3].value()
    assert g[3, 0] == -2 * x[2].value() + 4 * x[3].value() - 2 * x[4].value()
    assert g[4, 0] == -2 * x[3].value() + 2 * x[4].value()

    H = Hessian(y, x)

    expected_H = np.array(
        [
            [2.0, -2.0, 0.0, 0.0, 0.0],
            [-2.0, 4.0, -2.0, 0.0, 0.0],
            [0.0, -2.0, 4.0, -2.0, 0.0],
            [0.0, 0.0, -2.0, 4.0, -2.0],
            [0.0, 0.0, 0.0, -2.0, 2.0],
        ]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_sum_of_squares():
    r = VariableMatrix(4)
    r.set_value(np.array([[25.0], [10.0], [5.0], [0.0]]))

    x = VariableMatrix(4)
    for i in range(4):
        x[i].set_value(0.0)

    J = sum((r[i] - x[i]) * (r[i] - x[i]) for i in range(4))
    H = Hessian(J, x)

    expected_H = np.diag([2.0] * 4)
    assert (H.get().value() == expected_H).all()
    assert (H.value() == expected_H).all()


def test_nested_powers():
    x0 = 3.0

    x = Variable()
    x.set_value(x0)

    y = (x**2) ** 2

    J = Jacobian(y, x).value()
    assert J[0, 0] == pytest.approx(4 * x0 * x0 * x0, abs=1e-12)

    H = Hessian(y, x).value()
    assert H[0, 0] == pytest.approx(12 * x0 * x0, abs=1e-12)


def test_rosenbrock():
    # z = (1 − x)² + 100(y − x²)²
    #   = 100(−x² + y)² + (−x + 1)²
    #
    # ∂z/∂x = 200(−x² + y)⋅−2x + 2(−x + 1)⋅−1
    #       = −400x(−x² + y) − 2(−x + 1)
    #       = 400x³ − 400xy + 2x − 2
    #
    # ∂z/∂y = 200(−x² + y)
    #
    # ∂²z/∂x² = 1200x² − 400y + 2
    # ∂²z/∂xy = −400x
    # ∂²z/∂y² = 200

    input = VariableMatrix(2)
    x = input[0]
    y = input[1]
    hessian = Hessian((1 - x) ** 2 + 100 * (y - x**2) ** 2, input)

    for x0 in np.arange(-2.5, 2.5, 0.1):
        for y0 in np.arange(-2.5, 2.5, 0.1):
            x.set_value(x0)
            y.set_value(y0)

            H = hessian.value().todense()
            assert H[0, 0] == pytest.approx(1200 * x0**2 - 400 * y0 + 2, abs=1e-11)
            assert H[0, 1] == -400 * x0
            assert H[1, 0] == -400 * x0
            assert H[1, 1] == 200


def test_edge_pushing_wang_example_1():
    # See example 1 of [1]
    #
    # [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    #     efficient Hessian computation via automatic differentiation", 2016.
    #     https://sci-hub.st/10.1007/s12532-016-0100-3

    x = VariableMatrix(2)
    x[0].set_value(3)
    x[1].set_value(4)

    # y = (x₀sin(x₁)) x₀
    y = (x[0] * autodiff.sin(x[1])) * x[0]

    # dy/dx = [2x₀sin(x₁)  x₀²cos(x₁)]
    # dy/dx = [ 6sin(4)     9cos(4)  ]
    J = Jacobian(y, x)
    expected_J = np.array([[6.0 * math.sin(4.0), 9.0 * math.cos(4.0)]])
    assert (J.get().value() == expected_J).all()
    assert (J.value() == expected_J).all()

    #           [ 2sin(x₁)    2x₀cos(x₁)]
    # d²y/dx² = [2x₀cos(x₁)  −x₀²sin(x₁)]
    #
    #           [2sin(4)   6cos(4)]
    # d²y/dx² = [6cos(4)  −9sin(4)]
    H = Hessian(y, x)
    expected_H = np.array(
        [
            [2.0 * math.sin(4.0), 6.0 * math.cos(4.0)],
            [6.0 * math.cos(4.0), -9.0 * math.sin(4.0)],
        ]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value() == expected_H).all()


def test_edge_pushing_petro_figure_1():
    # See figure 1 of [1]
    #
    # [1] Petro, C. G., et al. "On efficient Hessian computation using the edge
    #     pushing algorithm in Julia", 2017.
    #     https://mlubin.github.io/pdf/edge_pushing_julia.pdf

    # y = p₁ log(x₁x₂)
    p_1 = Variable(2.0)
    x = VariableMatrix(2)
    x[0].set_value(2)
    x[1].set_value(3)
    y = p_1 * autodiff.log(x[0] * x[1])

    # d²y/dx² = [−p₁/x₁²     0   ]
    #           [   0     −p₁/x₂²]
    H = Hessian(y, x)
    expected_H = np.diag(
        [-p_1.value() / x[0].value() ** 2, -p_1.value() / x[1].value() ** 2]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value() == expected_H).all()


def test_variable_reuse():
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
