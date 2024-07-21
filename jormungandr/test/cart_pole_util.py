import math

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import VariableMatrix, Variable
import numpy as np

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
    qddot[2:, :] = np.linalg.solve(M, tau_g - C @ qdot + B @ u)
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
