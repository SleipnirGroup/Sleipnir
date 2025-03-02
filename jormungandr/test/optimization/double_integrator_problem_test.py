import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem


def test_double_integrator_problem():
    T = 3.5
    dt = 0.005
    N = int(T / dt)

    r = 2.0

    problem = Problem()

    # 2x1 state vector with N + 1 timesteps (includes last state)
    X = problem.decision_variable(2, N + 1)

    # 1x1 input vector with N timesteps (input at last state doesn't matter)
    U = problem.decision_variable(1, N)

    # Kinematics constraint assuming constant acceleration between timesteps
    for k in range(N):
        t = dt
        p_k1 = X[0, k + 1]
        v_k1 = X[1, k + 1]
        p_k = X[0, k]
        v_k = X[1, k]
        a_k = U[0, k]

        # pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
        problem.subject_to(p_k1 == p_k + v_k * t + 0.5 * a_k * t**2)

        # vₖ₊₁ = vₖ + aₖt
        problem.subject_to(v_k1 == v_k + a_k * t)

    # Start and end at rest
    problem.subject_to(X[:, :1] == np.array([[0.0], [0.0]]))
    problem.subject_to(X[:, N : N + 1] == np.array([[r], [0.0]]))

    # Limit velocity
    problem.subject_to(-1 <= X[1:2, :])
    problem.subject_to(X[1:2, :] <= 1)

    # Limit acceleration
    problem.subject_to(-1 <= U)
    problem.subject_to(U <= 1)

    # Cost function - minimize position error
    J = 0.0
    for k in range(N + 1):
        J += (r - X[0, k]) ** 2
    problem.minimize(J)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(0.0, abs=1e-8)
    assert X.value(1, 0) == pytest.approx(0.0, abs=1e-8)

    # Verify solution
    x = np.zeros((2, 1))
    u = np.zeros((1, 1))
    for k in range(N):
        # Verify state
        assert X.value(0, k) == pytest.approx(x[0, 0], abs=1e-2)
        assert X.value(1, k) == pytest.approx(x[1, 0], abs=1e-2)

        # Determine expected input for this timestep
        if k * dt < 1.0:
            # Accelerate
            u[0, 0] = 1.0
        elif k * dt < 2.05:
            # Maintain speed
            u[0, 0] = 0.0
        elif k * dt < 3.275:
            # Decelerate
            u[0, 0] = -1.0
        else:
            # Accelerate
            u[0, 0] = 1.0

        # Verify input
        if (
            k > 0
            and k < N - 1
            and abs(U.value(0, k - 1) - U.value(0, k + 1)) >= 1.0 - 1e-2
        ):
            # If control input is transitioning between -1, 0, or 1, ensure it's
            # within (-1, 1)
            assert U.value(0, k) >= -1.0
            assert U.value(0, k) <= 1.0
        else:
            assert U.value(0, k) == pytest.approx(u[0, 0], abs=1e-4)

        # Project state forward
        x = A @ x + B @ u

    # Verify final state
    assert X.value(0, N) == pytest.approx(r, abs=1e-8)
    assert X.value(1, N) == pytest.approx(0.0, abs=1e-8)

    # Log states for offline viewing
    with open("Double integrator states.csv", "w") as f:
        f.write("Time (s),Position (m),Velocity (rad/s)\n")

        for k in range(N + 1):
            f.write(f"{k * dt},{X.value(0, k)},{X.value(1, k)}\n")

    # Log inputs for offline viewing
    with open("Double integrator inputs.csv", "w") as f:
        f.write("Time (s),Acceleration (m/s²)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
