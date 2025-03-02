import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem


def test_maximize():
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(1.0)

    y = problem.decision_variable()
    y.set_value(1.0)

    problem.maximize(50 * x + 40 * y)

    problem.subject_to(x + 1.5 * y <= 750)
    problem.subject_to(2 * x + 3 * y <= 1500)
    problem.subject_to(2 * x + y <= 1000)
    problem.subject_to(x >= 0)
    problem.subject_to(y >= 0)

    assert problem.cost_function_type() == ExpressionType.LINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(375.0, abs=1e-6)
    assert y.value() == pytest.approx(250.0, abs=1e-6)


def test_free_variable():
    problem = Problem()

    x = problem.decision_variable(2)
    x[0].set_value(1.0)
    x[1].set_value(2.0)

    problem.subject_to(x[0] == 0)

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x[0].value() == pytest.approx(0.0, abs=1e-6)
    assert x[1].value() == pytest.approx(2.0, abs=1e-6)
