from jormungandr.autodiff import Gradient, Variable
import jormungandr.autodiff as autodiff

import math
import numpy as np
import pytest


def test_trivial_case():
    a = Variable()
    a.set_value(10)
    b = Variable()
    b.set_value(20)
    c = a

    assert 1 == Gradient(a, a).calculate()[0, 0]
    assert 0 == Gradient(a, b).calculate()[0, 0]
    assert 1 == Gradient(c, a).calculate()[0, 0]
    assert 0 == Gradient(c, b).calculate()[0, 0]


def test_positive_operator():
    a = Variable()
    a.set_value(10)
    c = +a

    assert a.value() == c.value()
    assert 1.0 == Gradient(c, a).calculate()[0, 0]


def test_negative_operator():
    a = Variable()
    a.set_value(10)
    c = -a

    assert -a.value() == c.value()
    assert -1.0 == Gradient(c, a).calculate()[0, 0]


def test_identical_variables():
    a = Variable()
    a.set_value(10)
    x = a
    c = a * a + x

    assert a.value() * a.value() + x.value() == c.value()
    assert (
        2 * a.value() + Gradient(x, a).calculate()[0, 0]
        == Gradient(c, a).calculate()[0, 0]
    )
    assert (
        2 * a.value() * Gradient(x, a).calculate()[0, 0] + 1
        == Gradient(c, a).calculate()[0, 0]
    )


def test_elementary():
    a = Variable()
    a.set_value(1.0)
    b = Variable()
    b.set_value(2.0)
    c = Variable()
    c.set_value(3.0)

    c = -2 * a
    assert -2 == Gradient(c, a).calculate()[0, 0]

    c = a / 3.0
    assert 1.0 / 3.0 == Gradient(c, a).calculate()[0, 0]

    a.set_value(100.0)
    b.set_value(200.0)

    c = a + b
    assert 1.0 == Gradient(c, a).calculate()[0, 0]
    assert 1.0 == Gradient(c, b).calculate()[0, 0]

    c = a - b
    assert 1.0 == Gradient(c, a).calculate()[0, 0]
    assert -1.0 == Gradient(c, b).calculate()[0, 0]

    c = -a + b
    assert -1.0 == Gradient(c, a).calculate()[0, 0]
    assert 1.0 == Gradient(c, b).calculate()[0, 0]

    c = a + 1
    assert 1.0 == Gradient(c, a).calculate()[0, 0]


def test_comparison():
    x = Variable()
    x.set_value(10.0)
    a = Variable()
    a.set_value(10.0)
    b = Variable()
    b.set_value(200.0)

    assert a.value() == a.value()
    assert a.value() == x.value()
    assert a.value() == 10
    assert 10 == a.value()

    assert a.value() != b.value()
    assert a.value() != 20
    assert 20 != a.value()

    assert a.value() < b.value()
    assert a.value() < 20

    assert b.value() > a.value()
    assert 20 > a.value()

    assert a.value() <= a.value()
    assert a.value() <= x.value()
    assert a.value() <= b.value()
    assert a.value() <= 10
    assert a.value() <= 20

    assert a.value() >= a.value()
    assert x.value() >= a.value()
    assert b.value() >= a.value()
    assert 10 >= a.value()
    assert 20 >= a.value()

    # Comparison between variables and expressions
    assert a.value() == a.value() / a.value() * a.value()
    assert a.value() / a.value() * a.value() == a.value()

    assert a.value() != (a - a).value()
    assert (a - a).value() != a.value()

    assert (a - a).value() < a.value()
    assert a.value() < (a + a).value()

    assert (a + a).value() > a.value()
    assert a.value() > (a - a).value()

    assert a.value() <= (a - a + a).value()
    assert (a - a + a).value() <= a.value()

    assert a.value() <= (a + a).value()
    assert (a - a).value() <= a.value()

    assert a.value() >= (a - a + a).value()
    assert (a - a + a).value() >= a.value()

    assert (a + a).value() >= a.value()
    assert a.value() >= (a - a).value()


def test_trigonometry():
    x = Variable()
    x.set_value(0.5)
    assert math.sin(x.value()) == autodiff.sin(x).value()
    assert math.cos(x.value()) == Gradient(autodiff.sin(x), x).calculate()[0, 0]

    assert math.cos(x.value()) == autodiff.cos(x).value()
    assert -math.sin(x.value()) == Gradient(autodiff.cos(x), x).calculate()[0, 0]

    assert math.tan(x.value()) == autodiff.tan(x).value()
    assert (
        1.0 / (math.cos(x.value()) * math.cos(x.value()))
        == Gradient(autodiff.tan(x), x).calculate()[0, 0]
    )

    assert math.asin(x.value()) == autodiff.asin(x).value()
    assert (
        1.0 / math.sqrt(1 - x.value() * x.value())
        == Gradient(autodiff.asin(x), x).calculate()[0, 0]
    )

    assert math.acos(x.value()) == autodiff.acos(x).value()
    assert (
        -1.0 / math.sqrt(1 - x.value() * x.value())
        == Gradient(autodiff.acos(x), x).calculate()[0, 0]
    )

    assert math.atan(x.value()) == autodiff.atan(x).value()
    assert (
        1.0 / (1 + x.value() * x.value())
        == Gradient(autodiff.atan(x), x).calculate()[0, 0]
    )


def test_hyperbolic():
    x = Variable()
    x.set_value(1.0)
    assert math.sinh(x.value()) == autodiff.sinh(x).value()
    assert math.cosh(x.value()) == Gradient(autodiff.sinh(x), x).calculate()[0, 0]

    assert math.cosh(x.value()) == autodiff.cosh(x).value()
    assert math.sinh(x.value()) == Gradient(autodiff.cosh(x), x).calculate()[0, 0]

    assert math.tanh(x.value()) == autodiff.tanh(x).value()
    assert (
        1.0 / (math.cosh(x.value()) * math.cosh(x.value()))
        == Gradient(autodiff.tanh(x), x).calculate()[0, 0]
    )


def test_exponential():
    x = Variable()
    x.set_value(1.0)
    assert math.log(x.value()) == autodiff.log(x).value()
    assert 1.0 / x.value() == Gradient(autodiff.log(x), x).calculate()[0, 0]

    assert math.log10(x.value()) == autodiff.log10(x).value()
    assert (
        1.0 / (math.log(10) * x.value())
        == Gradient(autodiff.log10(x), x).calculate()[0, 0]
    )

    assert math.exp(x.value()) == autodiff.exp(x).value()
    assert math.exp(x.value()) == Gradient(autodiff.exp(x), x).calculate()[0, 0]


def test_power():
    x = Variable()
    x.set_value(1.0)
    a = Variable()
    a.set_value(2.0)
    y = 2 * a
    assert math.sqrt(x.value()) == autodiff.sqrt(x).value()
    assert 0.5 / math.sqrt(x.value()) == Gradient(autodiff.sqrt(x), x).calculate()[0, 0]

    assert math.pow(x.value(), 2.0) == autodiff.pow(x, 2.0).value()
    assert 2.0 * x.value() == Gradient(autodiff.pow(x, 2.0), x).calculate()[0, 0]

    assert math.pow(2.0, x.value()) == autodiff.pow(2.0, x).value()
    assert (
        math.log(2.0) * math.pow(2.0, x.value())
        == Gradient(autodiff.pow(2.0, x), x).calculate()[0, 0]
    )

    assert math.pow(x.value(), x.value()) == autodiff.pow(x, x).value()
    assert ((autodiff.log(x) + 1) * autodiff.pow(x, x)).value() == Gradient(
        autodiff.pow(x, x), x
    ).calculate()[0, 0]

    assert 2 * a.value() == y.value()
    assert 2.0 == Gradient(y, a).calculate()[0, 0]

    assert math.pow(x.value(), y.value()) == autodiff.pow(x, y).value()
    assert (
        y.value() / x.value() * math.pow(x.value(), y.value())
        == Gradient(autodiff.pow(x, y), x).calculate()[0, 0]
    )
    assert (
        math.pow(x.value(), y.value())
        * (
            y.value() / x.value() * Gradient(x, a).calculate()[0, 0]
            + math.log(x.value()) * Gradient(y, a).calculate()[0, 0]
        )
        == Gradient(autodiff.pow(x, y), a).calculate()[0, 0]
    )
    assert (
        math.log(x.value()) * math.pow(x.value(), y.value())
        == Gradient(autodiff.pow(x, y), y).calculate()[0, 0]
    )


def test_abs():
    x = Variable()
    x.set_value(1.0)
    assert abs(x.value()) == autodiff.abs(x).value()
    assert 1.0 == Gradient(autodiff.abs(x), x).calculate()[0, 0]
    x.set_value(-1.0)
    assert abs(x.value()) == autodiff.abs(x).value()
    assert -1.0 == Gradient(autodiff.abs(x), x).calculate()[0, 0]
    x.set_value(0.0)
    assert abs(x.value()) == autodiff.abs(x).value()
    assert 0.0 == Gradient(autodiff.abs(x), x).calculate()[0, 0]


def test_atan2():
    # Testing atan2 function on (double, var)
    x = Variable()
    x.set_value(1.0)
    y = Variable()
    y.set_value(0.9)
    assert autodiff.atan2(2.0, x).value() == math.atan2(2.0, x.value())
    assert (
        Gradient(autodiff.atan2(2.0, x), x).calculate()[0, 0]
        == (-2.0 / (2 * 2 + x * x)).value()
    )

    # Testing atan2 function on (var, double)
    x.set_value(1.0)
    assert autodiff.atan2(x, 2.0).value() == math.atan2(x.value(), 2.0)
    assert (
        Gradient(autodiff.atan2(x, 2.0), x).calculate()[0, 0]
        == (2.0 / (2 * 2 + x * x)).value()
    )

    # Testing atan2 function on (var, var)
    x.set_value(1.1)
    assert autodiff.atan2(y, x).value() == math.atan2(y.value(), x.value())
    assert Gradient(autodiff.atan2(y, x), y).calculate()[0, 0] == pytest.approx(
        (x / (x * x + y * y)).value(), 1e-12
    )
    assert Gradient(autodiff.atan2(y, x), x).calculate()[0, 0] == pytest.approx(
        (-y / (x * x + y * y)).value(), 1e-12
    )

    # Testing atan2 function on (expr, expr)
    assert 3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1).value() == 3 * math.atan2(
        autodiff.sin(y).value(), 2 * x.value() + 1
    )
    assert Gradient(3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1), y).calculate()[
        0, 0
    ] == pytest.approx(
        (
            3
            * (2 * x + 1)
            * autodiff.cos(y)
            / ((2 * x + 1) * (2 * x + 1) + autodiff.sin(y) * autodiff.sin(y))
        ).value()
    )
    assert (
        Gradient(3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1), x).calculate()[0, 0]
        == (
            3
            * -2
            * autodiff.sin(y)
            / ((2 * x + 1) * (2 * x + 1) + autodiff.sin(y) * autodiff.sin(y))
        ).value()
    )


def test_hypot():
    # Testing hypot function on (var, double)
    x = Variable()
    x.set_value(1.8)
    y = Variable()
    y.set_value(1.5)
    assert math.hypot(x.value(), 2.0) == autodiff.hypot(x, 2.0).value()
    assert (x / math.hypot(x.value(), 2.0)).value() == Gradient(
        autodiff.hypot(x, 2.0), x
    ).calculate()[0, 0]

    # Testing hypot function on (double, var)
    assert math.hypot(2.0, y.value()) == autodiff.hypot(2.0, y).value()
    assert (y / math.hypot(2.0, y.value())).value() == Gradient(
        autodiff.hypot(2.0, y), y
    ).calculate()[0, 0]

    # Testing hypot function on (var, var)
    x.set_value(1.3)
    y.set_value(2.3)
    assert math.hypot(x.value(), y.value()) == autodiff.hypot(x, y).value()
    assert (x / math.hypot(x.value(), y.value())).value() == Gradient(
        autodiff.hypot(x, y), x
    ).calculate()[0, 0]
    assert (y / math.hypot(x.value(), y.value())).value() == Gradient(
        autodiff.hypot(x, y), y
    ).calculate()[0, 0]

    # Testing hypot function on (expr, expr)
    x.set_value(1.3)
    y.set_value(2.3)
    assert (
        math.hypot(2.0 * x.value(), 3.0 * y.value())
        == autodiff.hypot(2.0 * x, 3.0 * y).value()
    )
    assert (4.0 * x / math.hypot(2.0 * x.value(), 3.0 * y.value())).value() == Gradient(
        autodiff.hypot(2.0 * x, 3.0 * y), x
    ).calculate()[0, 0]
    assert (9.0 * y / math.hypot(2.0 * x.value(), 3.0 * y.value())).value() == Gradient(
        autodiff.hypot(2.0 * x, 3.0 * y), y
    ).calculate()[0, 0]

    # Testing hypot function on (var, var, var)
    x.set_value(1.3)
    y.set_value(2.3)
    z = Variable()
    z.set_value(3.3)
    assert (
        math.hypot(x.value(), y.value(), z.value()) == autodiff.hypot(x, y, z).value()
    )
    assert (x / math.hypot(x.value(), y.value(), z.value())).value() == Gradient(
        autodiff.hypot(x, y, z), x
    ).calculate()[0, 0]
    assert (y / math.hypot(x.value(), y.value(), z.value())).value() == Gradient(
        autodiff.hypot(x, y, z), y
    ).calculate()[0, 0]
    assert (z / math.hypot(x.value(), y.value(), z.value())).value() == Gradient(
        autodiff.hypot(x, y, z), z
    ).calculate()[0, 0]


def test_miscellaneous():
    x = Variable()
    x.set_value(3.0)

    assert abs(x.value()) == autodiff.abs(x).value()
    assert 1.0 == Gradient(x, x).calculate()[0, 0]

    x.set_value(0.5)
    assert math.erf(x.value()) == autodiff.erf(x).value()
    assert (
        2 / math.sqrt(math.pi) * math.exp(-x.value() * x.value())
        == Gradient(autodiff.erf(x), x).calculate()[0, 0]
    )


def test_reuse():
    a = Variable()
    a.set_value(10)
    b = Variable()
    b.set_value(20)
    x = a * b

    gradient = Gradient(x, a)

    g = gradient.calculate()
    assert 20.0 == g[0]

    b.set_value(10)
    g = gradient.calculate()
    assert 10.0 == g[0]


def test_sign():
    def sign(x):
        if x < 0.0:
            return -1.0
        elif x == 0.0:
            return 0.0
        else:
            return 1.0

    x = Variable()
    x.set_value(1.0)
    assert sign(x.value()) == autodiff.sign(x).value()
    assert 0.0 == Gradient(autodiff.sign(x), x).calculate()[0, 0]
    x.set_value(-1.0)
    assert sign(x.value()) == autodiff.sign(x).value()
    assert 0.0 == Gradient(autodiff.sign(x), x).calculate()[0, 0]
    x.set_value(0.0)
    assert sign(x.value()) == autodiff.sign(x).value()
    assert 0.0 == Gradient(autodiff.sign(x), x).calculate()[0, 0]
