// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>
#include <print>

#include <Eigen/Core>
#include <sleipnir/optimization/OptimizationProblem.hpp>

// FRC 2022 shooter trajectory optimization
//
// This program finds the optimal initial launch velocity and launch angle for
// the 2022 FRC game's target.

constexpr double field_width = 8.2296;    // 27 ft
constexpr double field_length = 16.4592;  // 54 ft
constexpr double g = 9.806;               // m/s²

#ifndef RUNNING_TESTS
int main() {
  // Robot initial velocity
  double robot_initial_v_x = 0.2;   // ft/s
  double robot_initial_v_y = -0.2;  // ft/s
  double robot_initial_v_z = 0.0;   // ft/s

  constexpr double max_launch_velocity = 10.0;

  Eigen::Vector3d shooter{{field_length / 4.0}, {field_width / 4.0}, {1.2}};
  const auto& shooter_x = shooter(0, 0);
  const auto& shooter_y = shooter(1, 0);
  const auto& shooter_z = shooter(2, 0);

  constexpr Eigen::Vector3d target{
      {field_length / 2.0}, {field_width / 2.0}, {2.64}};
  const auto& target_x = target(0, 0);
  const auto& target_y = target(1, 0);
  const auto& target_z = target(2, 0);

  sleipnir::OptimizationProblem problem;

  // Set up duration decision variables
  constexpr int N = 10;
  auto T = problem.DecisionVariable();
  problem.SubjectTo(T >= 0);
  T.SetValue(1);
  auto dt = T / N;

  //     [x position]
  //     [y position]
  //     [z position]
  // x = [x velocity]
  //     [y velocity]
  //     [z velocity]
  auto X = problem.DecisionVariable(6, N);

  auto p_x = X.Row(0);
  auto p_y = X.Row(1);
  auto p_z = X.Row(2);
  auto v_x = X.Row(3);
  auto v_y = X.Row(4);
  auto v_z = X.Row(5);

  // Position initial guess is linear interpolation between start and end
  // position
  for (int k = 0; k < N; ++k) {
    p_x(k).SetValue(std::lerp(shooter_x, target_x, static_cast<double>(k) / N));
    p_y(k).SetValue(std::lerp(shooter_y, target_y, static_cast<double>(k) / N));
    p_z(k).SetValue(std::lerp(shooter_z, target_z, static_cast<double>(k) / N));
  }

  // Velocity initial guess is max launch velocity toward goal
  Eigen::Vector3d uvec_shooter_to_target = (target - shooter).normalized();
  for (int k = 0; k < N; ++k) {
    v_x(k).SetValue(max_launch_velocity * uvec_shooter_to_target(0, 0));
    v_y(k).SetValue(max_launch_velocity * uvec_shooter_to_target(1, 0));
    v_z(k).SetValue(max_launch_velocity * uvec_shooter_to_target(2, 0));
  }

  // Shooter initial position
  problem.SubjectTo(X.Block(0, 0, 3, 1) == shooter);

  // Require initial launch velocity is below max
  //
  //   √{v_x² + v_y² + v_z²) ≤ vₘₐₓ
  //   v_x² + v_y² + v_z² ≤ vₘₐₓ²
  problem.SubjectTo(v_x(0) * v_x(0) + v_y(0) * v_y(0) + v_z(0) * v_z(0) <=
                    max_launch_velocity * max_launch_velocity);

  // Require final position is in center of target circle
  problem.SubjectTo(p_x(N - 1) == target_x);
  problem.SubjectTo(p_y(N - 1) == target_y);
  problem.SubjectTo(p_z(N - 1) == target_z);

  // Require the final velocity is down
  problem.SubjectTo(v_z(N - 1) < 0.0);

  auto f = [&](sleipnir::VariableMatrix x) {
    // x' = x'
    // y' = y'
    // z' = z'
    // x" = −a_D(v_x)
    // y" = −a_D(v_y)
    // z" = −g − a_D(v_z)
    //
    // where a_D(v) = ½ρv² C_D A / m
    constexpr double rho = 1.204;  // kg/m³
    constexpr double C_D = 0.5;
    constexpr double A = std::numbers::pi * 0.3;
    constexpr double m = 2.0;  // kg
    auto a_D = [](auto v) { return 0.5 * rho * v * v * C_D * A / m; };

    auto v_x = x(3, 0) + robot_initial_v_x;
    auto v_y = x(4, 0) + robot_initial_v_y;
    auto v_z = x(5, 0) + robot_initial_v_z;
    return sleipnir::VariableMatrix{{v_x},       {v_y},       {v_z},
                                    {-a_D(v_x)}, {-a_D(v_y)}, {-g - a_D(v_z)}};
  };

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N - 1; ++k) {
    auto h = dt;
    auto x_k = X.Col(k);
    auto x_k1 = X.Col(k + 1);

    auto k1 = f(x_k);
    auto k2 = f(x_k + h / 2 * k1);
    auto k3 = f(x_k + h / 2 * k2);
    auto k4 = f(x_k + h * k3);
    problem.SubjectTo(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4));
  }

  // Minimize time to goal
  problem.Minimize(T);

  problem.Solve({.diagnostics = true});

  // Initial velocity vector
  Eigen::Vector3d v = X.Block(3, 0, 3, 1).Value();

  double launch_velocity = v.norm();
  std::println("Launch velocity = {:.03} ms", launch_velocity);

  double pitch = std::atan2(v(2), std::hypot(v(0), v(1)));
  std::println("Pitch = {:.03}°", pitch * 180.0 / std::numbers::pi);

  double yaw = std::atan2(v(1), v(0));
  std::println("Yaw = {:.03}°", yaw * 180.0 / std::numbers::pi);

  std::println("Total time = {:.03} s", T.Value());
  std::println("dt = {:.03} ms", dt.Value() * 1e3);
}
#endif
