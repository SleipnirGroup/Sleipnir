import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem
from jormungandr.test.differential_drive_util import (
    differential_drive_dynamics,
    differential_drive_dynamics_double,
)
from jormungandr.test.rk4 import rk4


def lerp(a, b, t):
    return a + t * (b - a)


def test_differential_drive_problem():
    T = 5.0  # s
    dt = 0.05  # s
    N = int(T / dt)

    u_max = 12.0  # V

    x_initial = np.zeros((5, 1))
    x_final = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]])

    problem = Problem()

    # x = [x, y, heading, left velocity, right velocity]ᵀ
    X = problem.decision_variable(5, N + 1)

    # Initial guess
    for k in range(N + 1):
        X[0, k].set_value(lerp(x_initial[0, 0], x_final[0, 0], k / N))
        X[1, k].set_value(lerp(x_initial[1, 0], x_final[1, 0], k / N))

    # u = [left voltage, right voltage]ᵀ
    U = problem.decision_variable(2, N)

    # Initial conditions
    problem.subject_to(X[:, :1] == x_initial)

    # Final conditions
    problem.subject_to(X[:, N : N + 1] == x_final)

    # Input constraints
    problem.subject_to(U >= -u_max)
    problem.subject_to(U <= u_max)

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2]
            == rk4(differential_drive_dynamics, X[:, k : k + 1], U[:, k : k + 1], dt)
        )

    # Minimize sum squared states and inputs
    J = 0.0
    for k in range(N):
        J += X[:, k : k + 1].T @ X[:, k : k + 1] + U[:, k : k + 1].T @ U[:, k : k + 1]
    problem.minimize(J)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONLINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=1e-8)
    assert X.value(1, 0) == pytest.approx(x_initial[1, 0], abs=1e-8)
    assert X.value(2, 0) == pytest.approx(x_initial[2, 0], abs=1e-8)
    assert X.value(3, 0) == pytest.approx(x_initial[3, 0], abs=1e-8)
    assert X.value(4, 0) == pytest.approx(x_initial[4, 0], abs=1e-8)

    # Verify solution
    x = np.zeros((5, 1))
    for k in range(N):
        u = U[:, k : k + 1].value()

        # Input constraints
        assert U[0, k].value() >= -u_max
        assert U[0, k].value() <= u_max
        assert U[1, k].value() >= -u_max
        assert U[1, k].value() <= u_max

        # Verify state
        assert X.value(0, k) == pytest.approx(x[0, 0], abs=1e-8)
        assert X.value(1, k) == pytest.approx(x[1, 0], abs=1e-8)
        assert X.value(2, k) == pytest.approx(x[2, 0], abs=1e-8)
        assert X.value(3, k) == pytest.approx(x[3, 0], abs=1e-8)
        assert X.value(4, k) == pytest.approx(x[4, 0], abs=1e-8)

        # Project state forward
        x = rk4(differential_drive_dynamics_double, x, u, dt)

    # Verify final state
    assert X.value(0, N) == pytest.approx(x_final[0, 0], abs=1e-8)
    assert X.value(1, N) == pytest.approx(x_final[1, 0], abs=1e-8)
    assert X.value(2, N) == pytest.approx(x_final[2, 0], abs=1e-8)
    assert X.value(3, N) == pytest.approx(x_final[3, 0], abs=1e-8)
    assert X.value(4, N) == pytest.approx(x_final[4, 0], abs=1e-8)

    # Log states for offline viewing
    with open("Differential drive states.csv", "w") as f:
        f.write(
            "Time (s),X position (m),Y position (m),Heading (rad),Left velocity (m/s),Right velocity (m/s)\n"
        )

        for k in range(N + 1):
            f.write(
                f"{k * dt},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)},{X.value(3, k)},{X.value(4, k)}\n"
            )

    # Log inputs for offline viewing
    with open("Differential drive inputs.csv", "w") as f:
        f.write("Time (s),Left voltage (V),Right voltage (V)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)},{U.value(1, k)}\n")
            else:
                f.write(f"{k * dt},0.0,0.0\n")
