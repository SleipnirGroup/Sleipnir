import math

import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition
from jormungandr.test.cart_pole_util import (
    cart_pole_dynamics,
    cart_pole_dynamics_double,
)
from jormungandr.test.rk4 import rk4


def lerp(a, b, t):
    return a + t * (b - a)


def test_optimization_problem_cart_pole():
    T = 5.0  # s
    dt = 0.05  # s
    N = int(T / dt)

    u_max = 20.0  # N
    d_max = 2.0  # m

    x_initial = np.zeros((4, 1))
    x_final = np.array([[1.0], [math.pi], [0.0], [0.0]])

    problem = OptimizationProblem()

    # x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
    X = problem.decision_variable(4, N + 1)

    # Initial guess
    for k in range(N + 1):
        X[0, k].set_value(lerp(x_initial[0, 0], x_final[0, 0], k / N))
        X[1, k].set_value(lerp(x_initial[1, 0], x_final[1, 0], k / N))

    # u = f_x
    U = problem.decision_variable(1, N)

    # Initial conditions
    problem.subject_to(X[:, :1] == x_initial)

    # Final conditions
    problem.subject_to(X[:, N : N + 1] == x_final)

    # Cart position constraints
    problem.subject_to(X[:1, :] >= 0.0)
    problem.subject_to(X[:1, :] <= d_max)

    # Input constraints
    problem.subject_to(U >= -u_max)
    problem.subject_to(U <= u_max)

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2]
            == rk4(cart_pole_dynamics, X[:, k : k + 1], U[:, k : k + 1], dt)
        )

    # Minimize sum squared inputs
    J = 0.0
    for k in range(N):
        J += U[:, k : k + 1].T @ U[:, k : k + 1]
    problem.minimize(J)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.QUADRATIC
    assert status.equality_constraint_type == ExpressionType.NONLINEAR
    assert status.inequality_constraint_type == ExpressionType.LINEAR
    assert status.exit_condition == SolverExitCondition.SUCCESS

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=1e-8)
    assert X.value(1, 0) == pytest.approx(x_initial[1, 0], abs=1e-8)
    assert X.value(2, 0) == pytest.approx(x_initial[2, 0], abs=1e-8)
    assert X.value(3, 0) == pytest.approx(x_initial[3, 0], abs=1e-8)

    # Verify solution
    for k in range(N):
        # Cart position constraints
        assert X[0, k] >= 0.0
        assert X[0, k] <= d_max

        # Input constraints
        assert U[0, k] >= -u_max
        assert U[0, k] <= u_max

        # Dynamics constraints
        expected_x_k1 = rk4(
            cart_pole_dynamics_double,
            X[:, k : k + 1].value(),
            U[:, k : k + 1].value(),
            dt,
        )
        actual_x_k1 = X[:, k + 1 : k + 2].value()
        for row in range(actual_x_k1.shape[0]):
            assert actual_x_k1[row, 0] == pytest.approx(expected_x_k1[row, 0], abs=1e-8)

    # Verify final state
    assert X.value(0, N) == pytest.approx(x_final[0, 0], abs=1e-8)
    assert X.value(1, N) == pytest.approx(x_final[1, 0], abs=1e-8)
    assert X.value(2, N) == pytest.approx(x_final[2, 0], abs=1e-8)
    assert X.value(3, N) == pytest.approx(x_final[3, 0], abs=1e-8)

    # Log states for offline viewing
    with open("Cart-pole states.csv", "w") as f:
        f.write(
            "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),Pole angular velocity (rad/s)\n"
        )

        for k in range(N + 1):
            f.write(
                f"{k * dt},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)},{X.value(3, k)}\n"
            )

    # Log inputs for offline viewing
    with open("Cart-pole inputs.csv", "w") as f:
        f.write("Time (s),Cart force (N)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
