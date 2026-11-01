import math

import numpy as np
import pytest
from flywheel_util import optimal_flywheel_trajectory
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds


def test_flywheel_problem() -> None:
    TOTAL_TIME = 5.0
    dt = 0.005
    N = int(TOTAL_TIME / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = math.exp(-dt)
    B = 1.0 - math.exp(-dt)

    problem = Problem()
    X = problem.decision_variable(1, N + 1)
    U = problem.decision_variable(1, N)

    # Dynamics constraint
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2] == A * X[:, k : k + 1] + B * U[:, k : k + 1]
        )

    # State and input constraints
    problem.subject_to(X[0, 0] == 0.0)
    problem.subject_to(bounds(-12, U, 12))

    # Cost function - minimize error
    r = np.array([[10.0]])
    problem.minimize(
        sum((r - X[:, k : k + 1]).T @ (r - X[:, k : k + 1]) for k in range(N + 1))
    )

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    X_ref, U_ref = optimal_flywheel_trajectory(A, B, r[0, 0], 12.0, N)

    # The cost depends on uₖ only through the Buₖ it adds to xₖ₊₁, so U is
    # determined 1/B times less precisely than X
    x_tolerance = 1e-4
    u_tolerance = x_tolerance / B

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(0.0, abs=1e-8)

    # Verify solution
    for k in range(N + 1):
        assert X.value(0, k) == pytest.approx(X_ref[k], abs=x_tolerance)
        if k < N:
            assert U.value(0, k) == pytest.approx(U_ref[k], abs=u_tolerance)

    # Log states for offline viewing
    with open("Flywheel states.csv", "w") as f:
        f.write("Time (s),Velocity (rad/s)\n")

        for k in range(N + 1):
            f.write(f"{k * dt},{X.value(0, k)}\n")

    # Log inputs for offline viewing
    with open("Flywheel inputs.csv", "w") as f:
        f.write("Time (s),Voltage (V)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
