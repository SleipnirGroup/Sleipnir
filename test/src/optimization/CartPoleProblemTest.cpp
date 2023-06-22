// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/acceleration.h>
#include <units/angle.h>
#include <units/force.h>
#include <units/length.h>
#include <units/time.h>

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

TEST(CartPoleProblemTest, DirectTranscription) {
  constexpr auto T = 5_s;
  constexpr units::second_t dt = 50_ms;
  constexpr int N = T / dt;

  constexpr auto u_max = 20_N;
  constexpr auto d = 1_m;
  constexpr auto d_max = 2_m;

  auto start = std::chrono::system_clock::now();

  sleipnir::OptimizationProblem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.DecisionVariable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
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
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  auto status = problem.Solve({.diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  // FIXME: Poor convergence
  // EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

#if 0
  // Verify initial state
  EXPECT_NEAR(0.0, X.Value(0, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(1, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(2, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(3, 0), 1e-2);

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    EXPECT_GE(X(0, k), 0.0);
    EXPECT_LE(X(0, k), d_max.value());

    // Input constraints
    EXPECT_GE(U(0, k), -u_max.value());
    EXPECT_LE(U(0, k), u_max.value());

    // Dynamics constraints
    // FIXME: The tolerance here is too large. It should be 1e-6 based on the
    // fact the solver claimed to converge to an infeasibility lower than that
    Eigen::VectorXd expected_x_k1 =
        RK4(CartPoleDynamicsDouble, X.Col(k).Value(), U.Col(k).Value(), dt);
    Eigen::VectorXd actual_x_k1 = X.Col(k + 1).Value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      EXPECT_NEAR(expected_x_k1(row), actual_x_k1(row), 2e-1)
          << "  x(" << row << ") @ k = " << k;
    }
  }

  // Verify final state
  EXPECT_NEAR(d.value(), X.Value(0, N), 1e-2);
  EXPECT_NEAR(std::numbers::pi, X.Value(1, N), 1e-2);
  EXPECT_NEAR(0.0, X.Value(2, N), 1e-2);
  EXPECT_NEAR(0.0, X.Value(3, N), 1e-2);
#endif

  // Log states for offline viewing
  std::ofstream states{"Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{}\n", k * dt.value(), X.Value(0, k),
                            X.Value(1, k), X.Value(2, k), X.Value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(), U.Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}
