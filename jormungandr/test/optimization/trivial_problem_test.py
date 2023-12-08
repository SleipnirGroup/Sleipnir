from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition
import numpy as np


def test_empty():
    problem = OptimizationProblem()

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.SUCCESS


def test_no_cost_unconstrained_1():
    problem = OptimizationProblem()

    X = problem.decision_variable(2, 3)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.SUCCESS

    for row in range(X.rows()):
        for col in range(X.cols()):
            assert X.value(row, col) == 0.0


def test_no_cost_unconstrained_2():
    problem = OptimizationProblem()

    X = problem.decision_variable(2, 3)
    X.set_value([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.SUCCESS

    for row in range(X.rows()):
        for col in range(X.cols()):
            assert X.value(row, col) == 1.0
