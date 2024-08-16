import platform

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import (
    OptimizationProblem,
    SolverExitCondition,
)

import numpy as np
import pytest


def test_quartic():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    x.set_value(20.0)

    problem.minimize(autodiff.pow(x, 4))

    problem.subject_to(x >= 1)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONLINEAR
    assert status.equality_constraint_type == ExpressionType.NONE
    assert status.inequality_constraint_type == ExpressionType.LINEAR
    assert status.exit_condition == SolverExitCondition.SUCCESS

    assert x.value() == pytest.approx(1.0, abs=1e-6)


def test_rosenbrock_with_cubic_and_line_constraint():
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    for x0 in np.arange(-1.5, 1.5, 0.1):
        for y0 in np.arange(-0.5, 2.5, 0.1):
            problem = OptimizationProblem()

            x = problem.decision_variable()
            x.set_value(x0)
            y = problem.decision_variable()
            y.set_value(y0)

            problem.minimize(
                autodiff.pow(1 - x, 2) + 100 * autodiff.pow(y - autodiff.pow(x, 2), 2)
            )

            problem.subject_to(autodiff.pow(x - 1, 3) - y + 1 <= 0)
            problem.subject_to(x + y - 2 <= 0)

            status = problem.solve()

            assert status.cost_function_type == ExpressionType.NONLINEAR
            assert status.equality_constraint_type == ExpressionType.NONE
            assert status.inequality_constraint_type == ExpressionType.NONLINEAR
            assert status.exit_condition == SolverExitCondition.SUCCESS

            # Local minimum at (0.0, 0.0)
            # Global minimum at (1.0, 1.0)
            assert x.value() == pytest.approx(
                0.0, abs=1e-2
            ) or x.value() == pytest.approx(1.0, abs=1e-2)
            assert y.value() == pytest.approx(
                0.0, abs=1e-2
            ) or y.value() == pytest.approx(1.0, abs=1e-2)


def test_rosenbrock_with_disk_constraint():
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    for x0 in np.arange(-1.5, 1.5, 0.1):
        for y0 in np.arange(-1.5, 1.5, 0.1):
            problem = OptimizationProblem()

            x = problem.decision_variable()
            x.set_value(x0)
            y = problem.decision_variable()
            y.set_value(y0)

            problem.minimize(
                autodiff.pow(1 - x, 2) + 100 * autodiff.pow(y - autodiff.pow(x, 2), 2)
            )

            problem.subject_to(autodiff.pow(x, 2) + autodiff.pow(y, 2) <= 2)

            status = problem.solve()

            assert status.cost_function_type == ExpressionType.NONLINEAR
            assert status.equality_constraint_type == ExpressionType.NONE
            assert status.inequality_constraint_type == ExpressionType.QUADRATIC
            assert status.exit_condition == SolverExitCondition.SUCCESS

            assert x.value() == pytest.approx(1.0, abs=1e-1)
            assert y.value() == pytest.approx(1.0, abs=1e-1)


def test_narrow_feasible_region():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    x.set_value(20.0)

    y = problem.decision_variable()
    y.set_value(50.0)

    problem.minimize(autodiff.sqrt(x * x + y * y))

    problem.subject_to(y == -x + 5.0)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.NONLINEAR
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.NONE

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # FIXME: Fails on macOS arm64 with "diverging iterates"
        assert status.exit_condition == SolverExitCondition.DIVERGING_ITERATES
        return
    else:
        assert status.exit_condition == SolverExitCondition.SUCCESS

    assert x.value() == pytest.approx(2.5, abs=1e-2)
    assert y.value() == pytest.approx(2.5, abs=1e-2)
