import math

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition
import numpy as np


def near(expected, actual, tolerance):
    return abs(expected - actual) < tolerance


def test_optimization_problem_flywheel():
    T = 5.0
    dt = 0.005
    N = int(T / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = np.array([[math.exp(-dt)]])
    B = np.array([[1.0 - math.exp(-dt)]])

    problem = OptimizationProblem()
    X = problem.decision_variable(1, N + 1)
    U = problem.decision_variable(1, N)

    # Dynamics constraint
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2] == A @ X[:, k : k + 1] + B @ U[:, k : k + 1]
        )

    # State and input constraints
    problem.subject_to(X[0, 0] == 0.0)
    problem.subject_to(-12 <= U)
    problem.subject_to(U <= 12)

    # Cost function - minimize error
    r = np.array([[10.0]])
    J = 0.0
    for k in range(N + 1):
        J += (r - X[:, k : k + 1]).T @ (r - X[:, k : k + 1])
    problem.minimize(J)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.QUADRATIC
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.LINEAR
    assert status.exit_condition == SolverExitCondition.SUCCESS

    # Voltage for steady-state velocity:
    #
    # rₖ₊₁ = Arₖ + Buₖ
    # uₖ = B⁺(rₖ₊₁ − Arₖ)
    # uₖ = B⁺(rₖ − Arₖ)
    # uₖ = B⁺(I − A)rₖ
    u_ss = np.linalg.solve(B, np.eye(A.shape[0]) - A) @ r

    # Verify initial state
    assert near(0.0, X.value(0, 0), 1e-8)

    # Verify solution
    x = np.array([[0.0]])
    u = np.array([[0.0]])
    for k in range(N):
        # Verify state
        assert near(x[0, 0], X.value(0, k), 1e-2)

        # Determine expected input for this timestep
        error = r[0, 0] - x[0, 0]
        if error > 1e-2:
            # Max control input until the reference is reached
            u[0, 0] = 12.0
        else:
            # Maintain speed
            u = u_ss

        # Verify input
        if (
            k > 0
            and k < N - 1
            and near(12.0, U.value(0, k - 1), 1e-2)
            and near(u_ss[0, 0], U.value(0, k + 1), 1e-2)
        ):
            # If control input is transitioning between 12 and u_ss, ensure it's
            # within (u_ss, 12)
            assert u[0, 0] >= u_ss[0, 0]
            assert u[0, 0] <= 12.0
        else:
            assert near(u[0, 0], U.value(0, k), 1e-4)

        # Project state forward
        x = A @ x + B @ u

    # Verify final state
    assert near(r[0, 0], X.value(0, N), 1e-7)

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
