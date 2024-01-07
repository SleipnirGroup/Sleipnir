// Copyright (c) Sleipnir contributors

#include "CartPoleUtil.hpp"

#include <units/acceleration.h>
#include <units/length.h>
#include <units/mass.h>

Eigen::Vector<double, 4> CartPoleDynamicsDouble(
    const Eigen::Vector<double, 4>& x, const Eigen::Vector<double, 1>& u) {
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
  constexpr double m_c = (5_kg).value();        // Cart mass
  constexpr double m_p = (0.5_kg).value();      // Pole mass
  constexpr double l = (0.5_m).value();         // Pole length
  constexpr double g = (9.806_mps_sq).value();  // Acceleration due to gravity

  Eigen::Vector<double, 2> q = x.segment(0, 2);
  Eigen::Vector<double, 2> qdot = x.segment(2, 2);
  double theta = q(1);
  double thetadot = qdot(1);

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  Eigen::Matrix<double, 2, 2> M;
  M(0, 0) = m_c + m_p;
  M(0, 1) = m_p * l * cos(theta);  // NOLINT
  M(1, 0) = m_p * l * cos(theta);  // NOLINT
  M(1, 1) = m_p * std::pow(l, 2);

  Eigen::Matrix<double, 2, 2> Minv;
  Minv(0, 0) = M(1, 1);
  Minv(0, 1) = -M(0, 1);
  Minv(1, 0) = -M(1, 0);
  Minv(1, 1) = M(0, 0);
  double detM = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  Minv /= detM;

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  Eigen::Matrix<double, 2, 2> C;
  C(0, 0) = 0;
  C(0, 1) = -m_p * l * thetadot * sin(theta);  // NOLINT
  C(1, 0) = 0;
  C(1, 1) = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  Eigen::Vector<double, 2> tau_g;
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  Eigen::Vector<double, 4> qddot;
  qddot.segment(0, 2) = qdot;
  qddot.segment(2, 2) = Minv * (tau_g - C * qdot + B * u);
  return qddot;
}

sleipnir::VariableMatrix CartPoleDynamics(const sleipnir::VariableMatrix& x,
                                          const sleipnir::VariableMatrix& u) {
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
  constexpr double m_c = (5_kg).value();        // Cart mass
  constexpr double m_p = (0.5_kg).value();      // Pole mass
  constexpr double l = (0.5_m).value();         // Pole length
  constexpr double g = (9.806_mps_sq).value();  // Acceleration due to gravity

  auto q = x.Segment(0, 2);
  auto qdot = x.Segment(2, 2);
  auto theta = q(1);
  auto thetadot = qdot(1);

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  sleipnir::VariableMatrix M{2, 2};
  M(0, 0) = m_c + m_p;
  M(0, 1) = m_p * l * cos(theta);  // NOLINT
  M(1, 0) = m_p * l * cos(theta);  // NOLINT
  M(1, 1) = m_p * std::pow(l, 2);

  sleipnir::VariableMatrix Minv{2, 2};
  Minv(0, 0) = M(1, 1);
  Minv(0, 1) = -M(0, 1);
  Minv(1, 0) = -M(1, 0);
  Minv(1, 1) = M(0, 0);
  auto detM = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  Minv /= detM;

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  sleipnir::VariableMatrix C{2, 2};
  C(0, 0) = 0;
  C(0, 1) = -m_p * l * thetadot * sin(theta);  // NOLINT
  C(1, 0) = 0;
  C(1, 1) = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  sleipnir::VariableMatrix tau_g{2};
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  sleipnir::VariableMatrix qddot{4};
  qddot.Segment(0, 2) = qdot;
  qddot.Segment(2, 2) = Minv * (tau_g - C * qdot + B * u);
  return qddot;
}
