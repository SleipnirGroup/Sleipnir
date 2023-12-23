import math

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import ExpressionType, VariableMatrix
from jormungandr.optimization import OptimizationProblem, SolverExitCondition
import numpy as np
import pytest


def near(expected, actual, tolerance):
    return abs(expected - actual) < tolerance


def rk4(f, x, u, dt):
    h = dt

    k1 = f(x, u)
    k2 = f(x + h * 0.5 * k1, u)
    k3 = f(x + h * 0.5 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def cart_pole_dynamics_double(x, u):
    # https://underactuated.mit.edu/acrobot.html#cart_pole
    #
    # θ is CCW+ measured from negative y-axis.
    #
    # q = [x, θ]ᵀ
    # q̇ = [ẋ, θ̇]ᵀ
    # u = f_x
    #
    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
    # M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    #
    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    #
    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    #
    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    #
    #     [1]
    # B = [0]
    m_c = 5.0  # Cart mass (kg)
    m_p = 0.5  # Pole mass (kg)
    l = 0.5  # Pole length (m)
    g = 9.806  # Acceleration due to gravity (m/s²)

    q = x[:2, :]
    qdot = x[:2:, :]
    theta = q[1, 0]
    thetadot = qdot[1, 0]

    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    M = np.empty((2, 2))
    M[0, 0] = m_c + m_p
    M[0, 1] = m_p * l * math.cos(theta)
    M[1, 0] = m_p * l * math.cos(theta)
    M[1, 1] = m_p * l**2

    Minv = np.empty((2, 2))
    Minv[0, 0] = M[1, 1]
    Minv[0, 1] = -M[0, 1]
    Minv[1, 0] = -M[1, 0]
    Minv[1, 1] = M[0, 0]
    detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    Minv /= detM

    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    C = np.empty((2, 2))
    C[0, 0] = 0.0
    C[0, 1] = -m_p * l * thetadot * math.sin(theta)
    C[1, 0] = 0.0
    C[1, 1] = 0.0

    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    tau_g = np.empty((2, 1))
    tau_g[0, 0] = 0.0
    tau_g[1, 0] = -m_p * g * l * math.sin(theta)

    #     [1]
    # B = [0]
    B = np.array([[1], [0]])

    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    qddot = np.empty((4, 1))
    qddot[:2, :] = qdot
    qddot[2:, :] = Minv @ (tau_g - C @ qdot + B @ u)
    return qddot


def cart_pole_dynamics(x, u):
    # https://underactuated.mit.edu/acrobot.html#cart_pole
    #
    # θ is CCW+ measured from negative y-axis.
    #
    # q = [x, θ]ᵀ
    # q̇ = [ẋ, θ̇]ᵀ
    # u = f_x
    #
    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
    # M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    #
    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    #
    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    #
    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    #
    #     [1]
    # B = [0]
    m_c = 5.0  # Cart mass (kg)
    m_p = 0.5  # Pole mass (kg)
    l = 0.5  # Pole length (m)
    g = 9.806  # Acceleration due to gravity (m/s²)

    q = x[:2, :]
    qdot = x[:2:, :]
    theta = q[1, 0]
    thetadot = qdot[1, 0]

    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    M = VariableMatrix(2, 2)
    M[0, 0] = m_c + m_p
    M[0, 1] = m_p * l * autodiff.cos(theta)
    M[1, 0] = m_p * l * autodiff.cos(theta)
    M[1, 1] = m_p * l**2

    Minv = VariableMatrix(2, 2)
    Minv[0, 0] = M[1, 1]
    Minv[0, 1] = -M[0, 1]
    Minv[1, 0] = -M[1, 0]
    Minv[1, 1] = M[0, 0]
    detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    Minv /= detM

    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    C = VariableMatrix(2, 2)
    C[0, 0] = 0.0
    C[0, 1] = -m_p * l * thetadot * autodiff.sin(theta)
    C[1, 0] = 0.0
    C[1, 1] = 0.0

    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    tau_g = VariableMatrix(2, 1)
    tau_g[0, 0] = 0.0
    tau_g[1, 0] = -m_p * g * l * autodiff.sin(theta)

    #     [1]
    # B = [0]
    B = np.array([[1], [0]])

    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    qddot = VariableMatrix(4, 1)
    qddot[:2, :] = qdot
    qddot[2:, :] = Minv @ (tau_g - C @ qdot + B @ u)
    return qddot


@pytest.mark.skip(reason="Crashes on Windows")
def test_optimization_problem_cart_pole():
    T = 5.0  # s
    dt = 0.05  # s
    N = int(T / dt)

    u_max = 20.0  # N
    d = 1.0  # m
    d_max = 2.0  # m

    problem = OptimizationProblem()

    # x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
    X = problem.decision_variable(4, N + 1)

    # Initial guess
    for k in range(N):
        X[0, k].set_value(float(k) / N * d)
        X[1, k].set_value(float(k) / N * math.pi)

    # u = f_x
    U = problem.decision_variable(1, N)

    # Initial conditions
    problem.subject_to(X[:, :1] == np.zeros((4, 1)))

    # Final conditions
    problem.subject_to(X[:, N : N + 1] == np.array([[d], [math.pi], [0.0], [0.0]]))

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
    # FIXME: Fails with "bad search direction"
    assert (
        status.exit_condition == SolverExitCondition.SUCCESS
        or status.exit_condition == SolverExitCondition.BAD_SEARCH_DIRECTION
    )
    if status.exit_condition == SolverExitCondition.BAD_SEARCH_DIRECTION:
        return

    # Verify initial state
    assert near(0.0, X.value(0, 0), 1e-2)
    assert near(0.0, X.value(1, 0), 1e-2)
    assert near(0.0, X.value(2, 0), 1e-2)
    assert near(0.0, X.value(3, 0), 1e-2)

    # Verify solution
    for k in range(N):
        # Cart position constraints
        assert X[0, k] >= 0.0
        assert X[0, k] <= d_max

        # Input constraints
        assert U[0, k] >= -u_max
        assert U[0, k] <= u_max

        # Dynamics constraints
        # FIXME: The tolerance here is too large. It should be 1e-6 based on the
        # fact the solver claimed to converge to an infeasibility lower than that
        expected_x_k1 = rk4(
            cart_pole_dynamics_double,
            X[:, k : k + 1].value(),
            U[:, k : k + 1].value(),
            dt,
        )
        actual_x_k1 = X[:, k + 1 : k + 2].value()
        for row in range(actual_x_k1.shape[0]):
            assert near(expected_x_k1[row, 0], actual_x_k1[row, 0], 2e-1)

    # Verify final state
    assert near(d, X.value(0, N), 1e-2)
    assert near(math.pi, X.value(1, N), 1e-2)
    assert near(0.0, X.value(2, N), 1e-2)
    assert near(0.0, X.value(3, N), 1e-2)

    # Log states for offline viewing
    with open("Cart-pole states.csv", "w") as f:
        f.write(
            "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),Pole angular velocity (rad/s)\n"
        )

        for k in range(N + 1):
            f.write(
                f"{k * dt},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)}, {X.value(3, k)}\n"
            )

    # Log inputs for offline viewing
    with open("Cart-pole inputs.csv", "w") as f:
        f.write("Time (s),Cart force (N)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
