// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/acceleration.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/length.h>
#include <units/time.h>
#include <units/velocity.h>

#include "CmdlineArguments.hpp"

// This problem tests the case where regularization fails
TEST(ArmOnElevatorProblemTest, DirectTranscription) {
  constexpr int N = 800;

  constexpr auto kElevatorStartHeight = 1_m;
  constexpr auto kElevatorEndHeight = 1.25_m;
  constexpr auto kElevatorMaxVelocity = 1_mps;
  constexpr auto kElevatorMaxAcceleration = 2_mps_sq;

  constexpr auto kArmLength = 1_m;
  constexpr units::radian_t kArmStartAngle = 0_deg;
  constexpr units::radian_t kArmEndAngle = 180_deg;
  constexpr units::radians_per_second_t kArmMaxVelocity = 360_deg_per_s;
  constexpr units::radians_per_second_squared_t kArmMaxAcceleration =
      720_deg_per_s_sq;

  constexpr auto kEndEffectorMaxHeight = 1.8_m;

  constexpr units::second_t kTotalTime = 4_s;
  constexpr auto dt = kTotalTime / N;

  sleipnir::OptimizationProblem problem;

  auto elevator = problem.DecisionVariable(2, N + 1);
  auto elevatorAccel = problem.DecisionVariable(1, N);

  auto arm = problem.DecisionVariable(2, N + 1);
  auto armAccel = problem.DecisionVariable(1, N);

  for (int k = 0; k < N; ++k) {
    // Elevator dynamics constraints
    problem.SubjectTo(elevator(0, k + 1) ==
                      elevator(0, k) + elevator(1, k) * dt.value());
    problem.SubjectTo(elevator(1, k + 1) ==
                      elevator(1, k) + elevatorAccel(0, k) * dt.value());

    // Arm dynamics constraints
    problem.SubjectTo(arm(0, k + 1) == arm(0, k) + arm(1, k) * dt.value());
    problem.SubjectTo(arm(1, k + 1) == arm(1, k) + armAccel(0, k) * dt.value());
  }

  // Elevator start and end conditions
  problem.SubjectTo(elevator.Col(0) ==
                    Eigen::Vector2d({kElevatorStartHeight.value(), 0.0}));
  problem.SubjectTo(elevator.Col(N) ==
                    Eigen::Vector2d({kElevatorEndHeight.value(), 0.0}));

  // Arm start and end conditions
  problem.SubjectTo(arm.Col(0) ==
                    Eigen::Vector2d({kArmStartAngle.value(), 0.0}));
  problem.SubjectTo(arm.Col(N) == Eigen::Vector2d({kArmEndAngle.value(), 0.0}));

  // Elevator velocity limits
  problem.SubjectTo(-kElevatorMaxVelocity.value() <= elevator.Row(1));
  problem.SubjectTo(elevator.Row(1) <= kElevatorMaxVelocity.value());

  // Elevator acceleration limits
  problem.SubjectTo(-kElevatorMaxAcceleration.value() <= elevatorAccel);
  problem.SubjectTo(elevatorAccel <= kElevatorMaxAcceleration.value());

  // Arm velocity limits
  problem.SubjectTo(-kArmMaxVelocity.value() <= arm.Row(1));
  problem.SubjectTo(arm.Row(1) <= kArmMaxVelocity.value());

  // Arm acceleration limits
  problem.SubjectTo(-kArmMaxAcceleration.value() <= armAccel);
  problem.SubjectTo(armAccel <= kArmMaxAcceleration.value());

  // Height limit
  problem.SubjectTo(elevator.Row(0) +
                        kArmLength.value() *
                            arm.Row(0).CwiseTransform(sleipnir::sin) <=
                    kEndEffectorMaxHeight.value());

  // Cost function
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += sleipnir::pow(kElevatorEndHeight.value() - elevator(0, k), 2) +
         sleipnir::pow(kArmEndAngle.value() - arm(0, k), 2);
  }
  problem.Minimize(J);

  auto status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
            status.inequalityConstraintType);
  // FIXME: Fails with "bad search direction"
  // EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);
}
