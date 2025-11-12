import numpy as np
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem


def test_empty():
    problem = Problem()

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS


def test_no_cost_unconstrained_1():
    problem = Problem()

    X = problem.decision_variable(2, 3)

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    for row in range(X.rows()):
        for col in range(X.cols()):
            assert X.value(row, col) == 0.0


def test_no_cost_unconstrained_2():
    problem = Problem()

    X = problem.decision_variable(2, 3)
    X.set_value(np.ones((2, 3)))

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    for row in range(X.rows()):
        for col in range(X.cols()):
            assert X.value(row, col) == 1.0
