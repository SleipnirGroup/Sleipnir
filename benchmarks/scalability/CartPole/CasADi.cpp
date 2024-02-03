// Copyright (c) Sleipnir contributors

#include "CasADi.hpp"

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

casadi::MX CartPoleDynamics(const casadi::MX& x, const casadi::MX& u) {
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
  // τ_g(q) = [−m_p gl sinθ]
  //
  //     [1]
  // B = [0]
  constexpr double m_c = (5_kg).value();        // Cart mass
  constexpr double m_p = (0.5_kg).value();      // Pole mass
  constexpr double l = (0.5_m).value();         // Pole length
  constexpr double g = (9.806_mps_sq).value();  // Acceleration due to gravity

  auto q = x(casadi::Slice{0, 2});
  auto qdot = x(casadi::Slice{2, 4});
  auto theta = q(1);
  auto thetadot = qdot(1);

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  casadi::MX M{2, 2};
  M(0, 0) = m_c + m_p;
  M(0, 1) = m_p * l * cos(theta);  // NOLINT
  M(1, 0) = m_p * l * cos(theta);  // NOLINT
  M(1, 1) = m_p * std::pow(l, 2);

  casadi::MX Minv{2, 2};
  Minv(0, 0) = M(1, 1);
  Minv(0, 1) = -M(0, 1);
  Minv(1, 0) = -M(1, 0);
  Minv(1, 1) = M(0, 0);
  auto detM = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  Minv /= detM;

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  casadi::MX C{2, 2};
  C(0, 0) = 0;
  C(0, 1) = -m_p * l * thetadot * sin(theta);  // NOLINT
  C(1, 0) = 0;
  C(1, 1) = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  casadi::MX tau_g{2, 1};
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  casadi::MX B{2, 1};
  B(0) = 1.0;
  B(1) = 0.0;

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  casadi::MX qddot{4, 1};
  qddot(casadi::Slice{0, 2}) = qdot;
  qddot(casadi::Slice{2, 4}) =
      mtimes(Minv, tau_g - mtimes(C, qdot) + mtimes(B, u));
  return qddot;
}

casadi::Opti CartPoleCasADi(units::second_t dt, int N) {
  constexpr auto u_max = 20_N;
  constexpr auto d_max = 2_m;

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  casadi::Opti problem;
  casadi::Slice all;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    problem.set_initial(X(0, k), std::lerp(x_initial(0), x_final(0),
                                           static_cast<double>(k) / N));
    problem.set_initial(X(1, k), std::lerp(x_initial(1), x_final(1),
                                           static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.variable(1, N);

  // Initial conditions
  problem.subject_to(X(0, 0) == x_initial(0));
  problem.subject_to(X(1, 0) == x_initial(1));
  problem.subject_to(X(2, 0) == x_initial(2));
  problem.subject_to(X(3, 0) == x_initial(3));

  // Final conditions
  problem.subject_to(X(0, N) == x_final(0));
  problem.subject_to(X(1, N) == x_final(1));
  problem.subject_to(X(2, N) == x_final(2));
  problem.subject_to(X(3, N) == x_final(3));

  // Cart position constraints
  problem.subject_to(X(0, all) >= 0.0);
  problem.subject_to(X(0, all) <= d_max.value());

  // Input constraints
  problem.subject_to(U >= -u_max.value());
  problem.subject_to(U <= u_max.value());

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X(all, k + 1) ==
                       RK4<decltype(CartPoleDynamics), casadi::MX, casadi::MX>(
                           CartPoleDynamics, X(all, k), U(all, k), dt));
  }

  // Minimize sum squared inputs
  casadi::MX J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U(all, k).T() * U(all, k);
  }
  problem.minimize(J);

  return problem;
}
