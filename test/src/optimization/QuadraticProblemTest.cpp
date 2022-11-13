// Copyright (c) Joshua Nichols and Tyler Veness

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

TEST(QuadraticProblemTest, Unconstrained1d) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x = 2.0;

  problem.Minimize(x * x - 6.0 * x);

  sleipnir::SolverConfig config;
  config.diagnostics = true;

  auto status = problem.Solve(config);

  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
            status.costFunctionType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

  EXPECT_NEAR(3.0, x.Value(0), 1e-6);
}

TEST(QuadraticProblemTest, Unconstrained2d) {
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    x = 1.0;
    auto y = problem.DecisionVariable();
    y = 2.0;

    problem.Minimize(x * x + y * y);

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

    EXPECT_NEAR(0.0, x.Value(0), 1e-6);
    EXPECT_NEAR(0.0, y.Value(0), 1e-6);
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable(2);
    x(0) = 1.0;
    x(1) = 2.0;

    problem.Minimize(x.Transpose() * x);

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

    EXPECT_NEAR(0.0, x.Value(0), 1e-6);
    EXPECT_NEAR(0.0, x.Value(1), 1e-6);
  }
}

TEST(QuadraticProblemTest, EqualityConstrained) {
  // Maximize xy subject to x + 3y = 36.
  //
  // Maximize f(x,y) = xy
  // subject to g(x,y) = x + 3y - 36 = 0
  //
  //         value func  constraint
  //              |          |
  //              v          v
  // L(x,y,λ) = f(x,y) - λg(x,y)
  // L(x,y,λ) = xy - λ(x + 3y - 36)
  // L(x,y,λ) = xy - xλ - 3yλ + 36λ
  //
  // ∇_x,y,λ L(x,y,λ) = 0
  //
  // ∂L/∂x = y - λ
  // ∂L/∂y = x - 3λ
  // ∂L/∂λ = -x - 3y + 36
  //
  //  0x + 1y - 1λ = 0
  //  1x + 0y - 3λ = 0
  // -1x - 3y + 0λ + 36 = 0
  //
  // [ 0  1 -1][x]   [  0]
  // [ 1  0 -3][y] = [  0]
  // [-1 -3  0][λ]   [-36]
  //
  // Solve with:
  // ```python
  //   np.linalg.solve(
  //     np.array([[0,1,-1],
  //               [1,0,-3],
  //               [-1,-3,0]]),
  //     np.array([[0], [0], [-36]]))
  // ```
  //
  // [x]   [18]
  // [y] = [ 6]
  // [λ]   [ 6]
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    auto y = problem.DecisionVariable();

    problem.Maximize(x * y);

    problem.SubjectTo(x + 3 * y == 36);

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

    EXPECT_NEAR(18.0, x.Value(0), 1e-5);
    EXPECT_NEAR(6.0, y.Value(0), 1e-5);
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable(2);
    x(0) = 1.0;
    x(1) = 2.0;

    problem.Minimize(x.Transpose() * x);

    problem.SubjectTo(x == Eigen::Matrix<double, 2, 1>{{3.0, 3.0}});

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

    EXPECT_NEAR(3.0, x.Value(0), 1e-5);
    EXPECT_NEAR(3.0, x.Value(1), 1e-5);
  }
}
