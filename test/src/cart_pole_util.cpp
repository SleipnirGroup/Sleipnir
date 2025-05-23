// Copyright (c) Sleipnir contributors

#include "cart_pole_util.hpp"

#include <Eigen/QR>

// https://underactuated.mit.edu/acrobot.html#cart_pole
//
// θ is CCW+ measured from negative y-axis.
//
// q = [x, θ]ᵀ
// q̇ = [ẋ, θ̇]ᵀ
// u = f_x
//
// M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
// M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
// q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
//
//        [ m_c + m_p  m_p l cosθ]
// M(q) = [m_p l cosθ    m_p l²  ]
//
//           [0  −m_p lθ̇ sinθ]
// C(q, q̇) = [0       0      ]
//
//          [     0      ]
// τ_g(q) = [-m_p gl sinθ]
//
//     [1]
// B = [0]

inline constexpr double m_c = 5.0;  // Cart mass (kg)
inline constexpr double m_p = 0.5;  // Pole mass (kg)
inline constexpr double l = 0.5;    // Pole length (m)
inline constexpr double g = 9.806;  // Acceleration due to gravity (m/s²)

Eigen::Vector<double, 4> cart_pole_dynamics_double(
    const Eigen::Vector<double, 4>& x, const Eigen::Vector<double, 1>& u) {
  auto q = x.segment(0, 2);
  auto qdot = x.segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  Eigen::Matrix<double, 2, 2> M{
      {m_c + m_p, m_p * l * std::cos(theta)},
      {m_p * l * std::cos(theta), m_p * std::pow(l, 2)}};

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  Eigen::Matrix<double, 2, 2> C{{0, -m_p * l * thetadot * std::sin(theta)},
                                {0, 0}};

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  Eigen::Vector<double, 2> tau_g{{0}, {-m_p * g * l * std::sin(theta)}};

  //     [1]
  // B = [0]
  constexpr Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  Eigen::Vector<double, 4> qddot;
  qddot.segment(0, 2) = qdot;
  qddot.segment(2, 2) = M.householderQr().solve(tau_g - C * qdot + B * u);
  return qddot;
}

slp::VariableMatrix cart_pole_dynamics(const slp::VariableMatrix& x,
                                       const slp::VariableMatrix& u) {
  auto q = x.segment(0, 2);
  auto qdot = x.segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  slp::VariableMatrix M{{m_c + m_p, m_p * l * slp::cos(theta)},
                        {m_p * l * slp::cos(theta), m_p * std::pow(l, 2)}};

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  slp::VariableMatrix C{{0, -m_p * l * thetadot * slp::sin(theta)}, {0, 0}};

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  slp::VariableMatrix tau_g{{0}, {-m_p * g * l * slp::sin(theta)}};

  //     [1]
  // B = [0]
  constexpr Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  slp::VariableMatrix qddot{4};
  qddot.segment(0, 2) = qdot;
  qddot.segment(2, 2) = slp::solve(M, tau_g - C * qdot + B * u);
  return qddot;
}
