import numpy as np
import pytest
from sleipnir.autodiff import ExpressionType, Variable, VariableMatrix


def test_default_constructor() -> None:
    a = Variable()
    assert a.value() == 0.0
    assert a.type() == ExpressionType.LINEAR


def test_constant_constructor() -> None:
    # float
    a = Variable(1.0)
    assert a.value() == 1.0
    assert a.type() == ExpressionType.CONSTANT

    # int
    b = Variable(2)
    assert b.value() == 2
    assert b.type() == ExpressionType.CONSTANT


def test_set_value() -> None:
    a = Variable()

    # float
    a.set_value(1.0)
    assert a.value() == 1.0

    # int
    a.set_value(2)
    assert a.value() == 2


def test_numpy_array_operators() -> None:
    x = Variable(2.0)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    # float arrays
    for result, expected in [
        (x * A, [[2.0, 4.0], [6.0, 8.0]]),
        (A * x, [[2.0, 4.0], [6.0, 8.0]]),
        (A / x, [[0.5, 1.0], [1.5, 2.0]]),
    ]:
        assert isinstance(result, VariableMatrix)
        assert np.array_equal(result.value(), expected)

    # int arrays
    B = np.array([[1, 2], [3, 4]])
    for result, expected in [
        (x * B, [[2.0, 4.0], [6.0, 8.0]]),
        (B * x, [[2.0, 4.0], [6.0, 8.0]]),
        (B / x, [[0.5, 1.0], [1.5, 2.0]]),
    ]:
        assert isinstance(result, VariableMatrix)
        assert np.array_equal(result.value(), expected)

    # NumPy scalars
    a = np.float64(3.0)
    for result, expected in [
        (a + x, 5.0),
        (x + a, 5.0),
        (a - x, 1.0),
        (x - a, -1.0),
        (a * x, 6.0),
        (x * a, 6.0),
        (a / x, 1.5),
        (x / a, 2.0 / 3.0),
    ]:
        assert isinstance(result, Variable)
        assert result.value() == pytest.approx(expected)
    # mypy rejects this because NumPy's stubs don't let np.integer defer to
    # Variable.__rmul__, but at runtime NumPy calls Variable.__array_ufunc__
    assert isinstance(np.int64(3) * x, Variable)  # type: ignore[operator]
    assert isinstance(a <= x, type(Variable() <= 0.0))
    assert isinstance(x <= a, type(Variable() <= 0.0))

    # Other operations with NumPy arrays raise TypeError
    with pytest.raises(TypeError):
        _ = A + x
    with pytest.raises(TypeError):
        _ = A <= x
    with pytest.raises(TypeError):
        _ = np.array([1.0, 2.0]) * x
