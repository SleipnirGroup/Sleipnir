"""These tests ensure coverage of the off-nominal solver exit conditions"""

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition


def test_too_few_dofs():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x == 1)
    problem.subject_to(x == 2)
    problem.subject_to(y == 1)
    problem.subject_to(z == 1)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.TOO_FEW_DOFS


def test_locally_infeasible_equality_constraints():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x == y + 1)
    problem.subject_to(y == z + 1)
    problem.subject_to(z == x + 1)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.LOCALLY_INFEASIBLE


def test_locally_infeasible_inequality_constraints():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x >= y + 1)
    problem.subject_to(y >= z + 1)
    problem.subject_to(z >= x + 1)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONE
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.LINEAR
    assert status.exit_condition == SolverExitCondition.LOCALLY_INFEASIBLE


def test_max_iterations_exceeded():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    x.set(0.0)
    problem.minimize(x)

    status = problem.solve(max_iterations=0, diagnostics=True)

    assert status.cost_function_type == ExpressionType.LINEAR
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.MAX_ITERATIONS_EXCEEDED


def test_timeout():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    x.set(0.0)
    problem.minimize(x)

    status = problem.solve(timeout=0.0, diagnostics=True)

    assert status.cost_function_type == ExpressionType.LINEAR
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.NONE
    assert status.exit_condition == SolverExitCondition.MAX_WALL_CLOCK_TIME_EXCEEDED
