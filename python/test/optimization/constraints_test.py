import numpy as np
from sleipnir.autodiff import Matrix, Real
from sleipnir.optimization import EqualityConstraints, InequalityConstraints


def test_equality_constraint_boolean_comparisons() -> None:
    args = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    # double-Real
    for lhs, rhs in args:
        assert bool(float(lhs) == Real(rhs)) == (lhs == rhs)

    # double-Matrix
    for lhs, rhs in args:
        assert bool(float(lhs) == Matrix([[rhs]])) == (lhs == rhs)

    # Real-double
    for lhs, rhs in args:
        assert bool(Real(lhs) == float(rhs)) == (lhs == rhs)

    # Real-Real
    for lhs, rhs in args:
        assert bool(Real(lhs) == Real(rhs)) == (lhs == rhs)

    # Real-Matrix
    for lhs, rhs in args:
        assert bool(Real(lhs) == Matrix([[rhs]])) == (lhs == rhs)

    # Matrix-double
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) == float(rhs)) == (lhs == rhs)

    # Matrix-Real
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) == Real(rhs)) == (lhs == rhs)

    # Matrix-Matrix
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) == Matrix([[rhs]])) == (lhs == rhs)

    # np.array-Matrix
    for lhs, rhs in args:
        assert bool(np.array([[lhs]]) == Matrix([[rhs]])) == (lhs == rhs)

    # np.array-Submatrix
    for lhs, rhs in args:
        assert bool(np.array([[lhs]]) == Matrix([[rhs]])[:, :]) == (lhs == rhs)

    # Matrix-np.array
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) == np.array([[rhs]])) == (lhs == rhs)

    # Submatrix-np.array
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]])[:, :] == np.array([[rhs]])) == (lhs == rhs)


# The solver treats a < constraint the same as a <= constraint, but the
# boolean conversion distinguishes them
def test_inequality_constraint_boolean_comparisons() -> None:
    args = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    # double-Real
    for lhs, rhs in args:
        assert bool(float(lhs) < Real(rhs)) == (lhs < rhs)
        assert bool(float(lhs) <= Real(rhs)) == (lhs <= rhs)
        assert bool(float(lhs) > Real(rhs)) == (lhs > rhs)
        assert bool(float(lhs) >= Real(rhs)) == (lhs >= rhs)

    # double-Matrix
    for lhs, rhs in args:
        assert bool(float(lhs) < Matrix([[rhs]])) == (lhs < rhs)
        assert bool(float(lhs) <= Matrix([[rhs]])) == (lhs <= rhs)
        assert bool(float(lhs) > Matrix([[rhs]])) == (lhs > rhs)
        assert bool(float(lhs) >= Matrix([[rhs]])) == (lhs >= rhs)

    # Real-Real
    for lhs, rhs in args:
        assert bool(Real(lhs) < Real(rhs)) == (lhs < rhs)
        assert bool(Real(lhs) <= Real(rhs)) == (lhs <= rhs)
        assert bool(Real(lhs) > Real(rhs)) == (lhs > rhs)
        assert bool(Real(lhs) >= Real(rhs)) == (lhs >= rhs)

    # Real-Matrix
    for lhs, rhs in args:
        assert bool(Real(lhs) < Matrix([[rhs]])) == (lhs < rhs)
        assert bool(Real(lhs) <= Matrix([[rhs]])) == (lhs <= rhs)
        assert bool(Real(lhs) > Matrix([[rhs]])) == (lhs > rhs)
        assert bool(Real(lhs) >= Matrix([[rhs]])) == (lhs >= rhs)

    # Matrix-double
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) < float(rhs)) == (lhs < rhs)
        assert bool(Matrix([[lhs]]) <= float(rhs)) == (lhs <= rhs)
        assert bool(Matrix([[lhs]]) > float(rhs)) == (lhs > rhs)
        assert bool(Matrix([[lhs]]) >= float(rhs)) == (lhs >= rhs)

    # Matrix-Real
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) < Real(rhs)) == (lhs < rhs)
        assert bool(Matrix([[lhs]]) <= Real(rhs)) == (lhs <= rhs)
        assert bool(Matrix([[lhs]]) > Real(rhs)) == (lhs > rhs)
        assert bool(Matrix([[lhs]]) >= Real(rhs)) == (lhs >= rhs)

    # Matrix-Matrix
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) < Matrix([[rhs]])) == (lhs < rhs)
        assert bool(Matrix([[lhs]]) <= Matrix([[rhs]])) == (lhs <= rhs)
        assert bool(Matrix([[lhs]]) > Matrix([[rhs]])) == (lhs > rhs)
        assert bool(Matrix([[lhs]]) >= Matrix([[rhs]])) == (lhs >= rhs)

    # np.array-Matrix
    for lhs, rhs in args:
        assert bool(np.array([[lhs]]) < Matrix([[rhs]])) == (lhs < rhs)
        assert bool(np.array([[lhs]]) <= Matrix([[rhs]])) == (lhs <= rhs)
        assert bool(np.array([[lhs]]) > Matrix([[rhs]])) == (lhs > rhs)
        assert bool(np.array([[lhs]]) >= Matrix([[rhs]])) == (lhs >= rhs)

    # np.array-Submatrix
    for lhs, rhs in args:
        assert bool(np.array([[lhs]]) < Matrix([[rhs]])[:, :]) == (lhs < rhs)
        assert bool(np.array([[lhs]]) <= Matrix([[rhs]])[:, :]) == (lhs <= rhs)
        assert bool(np.array([[lhs]]) > Matrix([[rhs]])[:, :]) == (lhs > rhs)
        assert bool(np.array([[lhs]]) >= Matrix([[rhs]])[:, :]) == (lhs >= rhs)

    # Matrix-np.array
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]]) < np.array([[rhs]])) == (lhs < rhs)
        assert bool(Matrix([[lhs]]) <= np.array([[rhs]])) == (lhs <= rhs)
        assert bool(Matrix([[lhs]]) > np.array([[rhs]])) == (lhs > rhs)
        assert bool(Matrix([[lhs]]) >= np.array([[rhs]])) == (lhs >= rhs)

    # Submatrix-np.array
    for lhs, rhs in args:
        assert bool(Matrix([[lhs]])[:, :] < np.array([[rhs]])) == (lhs < rhs)
        assert bool(Matrix([[lhs]])[:, :] <= np.array([[rhs]])) == (lhs <= rhs)
        assert bool(Matrix([[lhs]])[:, :] > np.array([[rhs]])) == (lhs > rhs)
        assert bool(Matrix([[lhs]])[:, :] >= np.array([[rhs]])) == (lhs >= rhs)


def test_equality_constraint_concatenation() -> None:
    eq1 = Real(1.0) == Real(1.0)
    eq2 = Real(1.0) == Real(2.0)
    eqs = EqualityConstraints([eq1, eq2])

    assert bool(eq1)
    assert not bool(eq2)
    assert not bool(eqs)


def test_inequality_constraint_concatenation() -> None:
    ineq1 = Real(2.0) < Real(1.0)
    ineq2 = Real(1.0) < Real(2.0)
    ineqs = InequalityConstraints([ineq1, ineq2])

    assert not bool(ineq1)
    assert bool(ineq2)
    assert not bool(ineqs)

    # Each constraint keeps its own strictness
    strict = Real(1.0) < Real(1.0)
    non_strict = Real(1.0) <= Real(1.0)
    assert not bool(InequalityConstraints([strict, non_strict]))
    assert not bool(InequalityConstraints([non_strict, strict]))
    assert bool(InequalityConstraints([non_strict, non_strict]))
    assert bool(InequalityConstraints([ineq2, non_strict]))
