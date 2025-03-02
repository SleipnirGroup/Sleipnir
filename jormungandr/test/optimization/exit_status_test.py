"""These tests ensure coverage of the off-nominal exit statuses"""

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem


def test_callback_requested_stop():
    problem = Problem()

    x = problem.decision_variable()
    problem.minimize(x * x)

    problem.add_callback(lambda info: False)
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    problem.add_callback(lambda info: True)
    assert problem.solve(diagnostics=True) == ExitStatus.CALLBACK_REQUESTED_STOP

    problem.clear_callbacks()
    problem.add_callback(lambda info: False)
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS


def test_too_few_dofs():
    problem = Problem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x == 1)
    problem.subject_to(x == 2)
    problem.subject_to(y == 1)
    problem.subject_to(z == 1)

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.TOO_FEW_DOFS


def test_locally_infeasible_equality_constraints():
    problem = Problem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x == y + 1)
    problem.subject_to(y == z + 1)
    problem.subject_to(z == x + 1)

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.LOCALLY_INFEASIBLE


def test_locally_infeasible_inequality_constraints():
    problem = Problem()

    x = problem.decision_variable()
    y = problem.decision_variable()
    z = problem.decision_variable()

    problem.subject_to(x >= y + 1)
    problem.subject_to(y >= z + 1)
    problem.subject_to(z >= x + 1)

    assert problem.cost_function_type() == ExpressionType.NONE
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.LOCALLY_INFEASIBLE


def test_diverging_iterates():
    problem = Problem()

    x = problem.decision_variable()
    problem.minimize(x)

    assert problem.cost_function_type() == ExpressionType.LINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.DIVERGING_ITERATES


def test_max_iterations_exceeded():
    problem = Problem()

    x = problem.decision_variable()
    problem.minimize(x * x)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert (
        problem.solve(max_iterations=0, diagnostics=True)
        == ExitStatus.MAX_ITERATIONS_EXCEEDED
    )


def test_timeout():
    problem = Problem()

    x = problem.decision_variable()
    problem.minimize(x * x)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(timeout=0.0, diagnostics=True) == ExitStatus.TIMEOUT
