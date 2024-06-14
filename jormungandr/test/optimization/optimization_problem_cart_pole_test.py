import math

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import ExpressionType, VariableMatrix, Variable
from jormungandr.optimization import OptimizationProblem, SolverExitCondition
import numpy as np
import pytest


def rk4(f, x, u, dt):
    h = dt

    k1 = f(x, u)
    k2 = f(x + h * 0.5 * k1, u)
    k3 = f(x + h * 0.5 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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


def cart_pole_dynamics_double(x, u):
    q = x[:2, :]
    qdot = x[2:, :]
    theta = q[1, 0]
    thetadot = qdot[1, 0]

    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    M = np.array(
        [
            [m_c + m_p, m_p * l * math.cos(theta)],
            [m_p * l * math.cos(theta), m_p * l**2],
        ]
    )

    detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    Minv = np.array(
        [[M[1, 1] / detM, -M[0, 1] / detM], [-M[1, 0] / detM, M[0, 0] / detM]]
    )

    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    C = np.array([[0.0, -m_p * l * thetadot * math.sin(theta)], [0.0, 0.0]])

    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    tau_g = np.array([[0.0], [-m_p * g * l * math.sin(theta)]])

    #     [1]
    # B = [0]
    B = np.array([[1], [0]])

    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    qddot = np.empty((4, 1))
    qddot[:2, :] = qdot
    qddot[2:, :] = Minv @ (tau_g - C @ qdot + B @ u)
    return qddot


def cart_pole_dynamics(x, u):
    q = x[:2, :]
    qdot = x[2:, :]
    theta = q[1, 0]
    thetadot = qdot[1, 0]

    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]
    M = VariableMatrix(
        [
            [Variable(m_c + m_p), m_p * l * autodiff.cos(theta)],
            [m_p * l * autodiff.cos(theta), Variable(m_p * l**2)],
        ]
    )

    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]
    C = VariableMatrix(
        [
            [Variable(0.0), -m_p * l * thetadot * autodiff.sin(theta)],
            [Variable(0.0), Variable(0.0)],
        ]
    )

    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]
    tau_g = VariableMatrix([[Variable(0.0)], [-m_p * g * l * autodiff.sin(theta)]])

    #     [1]
    # B = [0]
    B = np.array([[1], [0]])

    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    qddot = VariableMatrix(4, 1)
    qddot[:2, :] = qdot
    qddot[2:, :] = autodiff.solve(M, tau_g - C @ qdot + B @ u)
    return qddot


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
