from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition

import pytest


def test_maximize():
    problem = OptimizationProblem()

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

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.LINEAR
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.LINEAR
    assert status.exit_condition == SolverExitCondition.SUCCESS

    assert x.value() == pytest.approx(375.0, abs=1e-6)
    assert y.value() == pytest.approx(250.0, abs=1e-6)
