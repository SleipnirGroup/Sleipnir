// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>
#include <sleipnir/optimization/ocp/timestep_method.hpp>
#include <sleipnir/optimization/ocp/transcription_method.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/util/function_ref.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "flywheel_util.hpp"
#include "scalar_types.hpp"

template <typename T>
void flywheel_test(
    std::string test_name, T A, T B,
    const slp::function_ref<slp::VariableMatrix<T>(
        const slp::VariableMatrix<T>& x, const slp::VariableMatrix<T>& u)>& f,
    slp::DynamicsType dynamics_type,
    slp::TranscriptionMethod transcription_method) {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.005)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  using std::exp;
  T A_discrete = exp(A * T(dt.count()));
  T B_discrete = (T(1) - A_discrete) * B;

  constexpr T r(10);

  slp::OCP<T> problem(1, 1, dt, N, f, dynamics_type, slp::TimestepMethod::FIXED,
                      transcription_method);
  problem.constrain_initial_state(T(0));
  problem.set_upper_input_bound(T(12));
  problem.set_lower_input_bound(T(-12));

  // Set up cost
  Eigen::Matrix<T, 1, N + 1> r_mat = Eigen::Matrix<T, 1, N + 1>::Constant(r);
  problem.minimize((r_mat - problem.X()) * (r_mat - problem.X()).T());

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  if (transcription_method == slp::TranscriptionMethod::SINGLE_SHOOTING) {
    // The initial state is set directly instead of constrained
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  } else {
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  }
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  auto [X_ref, U_ref] =
      optimal_flywheel_trajectory(A_discrete, B_discrete, r, T(12), N);

  // The cost depends on uₖ only through the B·uₖ it adds to xₖ₊₁, so U is
  // determined 1/B times less precisely than X
  const T x_tolerance = T(1e-4);
  const T u_tolerance = x_tolerance / B_discrete;

  // Verify initial state
  CHECK_THAT(problem.X().value(0, 0), WithinAbs(T(0), T(1e-8)));

  // Verify solution
  for (int k = 0; k < N + 1; ++k) {
    INFO(std::format("  k = {}", k));

    CHECK_THAT(problem.X().value(0, k), WithinAbs(X_ref(k), x_tolerance));
    if (k < N) {
      if (transcription_method ==
          slp::TranscriptionMethod::DIRECT_COLLOCATION) {
        // The tolerance is large because the trajectory is represented by a
        // spline, and splines chatter when transitioning quickly between
        // steady-states.
        CHECK_THAT(problem.U().value(0, k), WithinAbs(U_ref(k), T(2)));
      } else {
        CHECK_THAT(problem.U().value(0, k), WithinAbs(U_ref(k), u_tolerance));
      }
    }
  }

  // Log states for offline viewing
  std::ofstream states{std::format("{} states.csv", test_name)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{}\n", T(k) * dt.count(),
                            problem.X().value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{std::format("{} inputs.csv", test_name)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", T(k) * dt.count(),
                              problem.U().value(0, k));
      } else {
        inputs << std::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}

TEMPLATE_TEST_CASE("OCP - Flywheel (explicit)", "[OCP]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T A(-1);
  constexpr T B(1);

  auto f_ode = [=](slp::VariableMatrix<T> x, slp::VariableMatrix<T> u) {
    return A * x + B * u;
  };

  flywheel_test<T>("OCP - Flywheel (explicit) direct collocation", A, B, f_ode,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::DIRECT_COLLOCATION);
  flywheel_test<T>("OCP - Flywheel (explicit) direct transcription", A, B,
                   f_ode, slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test<T>("OCP - Flywheel (explicit) single-shooting", A, B, f_ode,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::SINGLE_SHOOTING);
}

TEMPLATE_TEST_CASE("OCP - Flywheel (discrete)", "[OCP]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T A(-1);
  constexpr T B(1);
  constexpr std::chrono::duration<T> dt{T(0.005)};

  using std::exp;
  T A_discrete = exp(A * T(dt.count()));
  T B_discrete = (T(1) - A_discrete) * B;

  auto f_discrete = [=](slp::VariableMatrix<T> x, slp::VariableMatrix<T> u) {
    return A_discrete * x + B_discrete * u;
  };

  flywheel_test<T>("OCP - Flywheel (discrete) direct transcription", A, B,
                   f_discrete, slp::DynamicsType::DISCRETE,
                   slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test<T>("OCP - Flywheel (discrete) single-shooting", A, B,
                   f_discrete, slp::DynamicsType::DISCRETE,
                   slp::TranscriptionMethod::SINGLE_SHOOTING);
}
