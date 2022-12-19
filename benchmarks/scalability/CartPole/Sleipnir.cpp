// Copyright (c) Sleipnir contributors

#include "Sleipnir.hpp"

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <units/acceleration.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/force.h>
#include <units/length.h>
#include <units/mass.h>
#include <units/voltage.h>

/**
 * Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.
 *
 * @param f  The function to integrate. It must take two arguments x and u.
 * @param x  The initial value of x.
 * @param u  The value u held constant over the integration period.
 * @param dt The time over which to integrate.
 */
template <typename F, typename T, typename U>
T RK4(F&& f, T x, U u, units::second_t dt) {
  const auto h = dt.value();

  T k1 = f(x, u);
  T k2 = f(x + h * 0.5 * k1, u);
  T k3 = f(x + h * 0.5 * k2, u);
  T k4 = f(x + h * k3, u);

  return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

sleipnir::VariableMatrix CartPoleDynamics(const sleipnir::VariableMatrix& x,
                                          const sleipnir::VariableMatrix& u) {
  // https://underactuated.mit.edu/acrobot.html#cart_pole
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
  constexpr double m_c = (1_kg).value();        // Cart mass
  constexpr double m_p = (0.3_kg).value();      // Pole mass
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
  sleipnir::VariableMatrix tau_g{2, 1};
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  sleipnir::VariableMatrix qddot{4, 1};
  qddot.Segment(0, 2) = qdot;
  qddot.Segment(2, 2) = Minv * (tau_g - C * qdot + B * u);
  return qddot;
}

sleipnir::OptimizationProblem CartPoleSleipnir(units::second_t dt, int N) {
  constexpr auto u_max = 20_N;
  constexpr auto d = 1_m;
  constexpr auto d_max = 2_m;

  sleipnir::OptimizationProblem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.DecisionVariable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    X(0, k) = static_cast<double>(k) / N * d.value();
    X(1, k) = static_cast<double>(k) / N * std::numbers::pi;
  }

  // u = f_x
  auto U = problem.DecisionVariable(1, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) ==
                    Eigen::Matrix<double, 4, 1>{0.0, 0.0, 0.0, 0.0});

  // Final conditions
  problem.SubjectTo(X.Col(N) == Eigen::Matrix<double, 4, 1>{
                                    d.value(), std::numbers::pi, 0.0, 0.0});

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max.value());

  // Input constraints
  problem.SubjectTo(U >= -u_max.value());
  problem.SubjectTo(U <= u_max.value());

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(CartPoleDynamics), sleipnir::VariableMatrix,
                          sleipnir::VariableMatrix>(CartPoleDynamics, X.Col(k),
                                                    U.Col(k), dt));
  }

  // Minimize sum squared inputs
  sleipnir::VariableMatrix J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  return problem;
}
