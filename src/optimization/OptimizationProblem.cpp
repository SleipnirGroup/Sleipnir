// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/OptimizationProblem.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <limits>
#include <string>

#include <fmt/core.h>

#include "Filter.hpp"
#include "RegularizedLDLT.hpp"
#include "ScopeExit.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/util/AutodiffUtil.hpp"
#include "sleipnir/util/SparseUtil.hpp"

using namespace sleipnir;

// Works cited:
//
// [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
//     Springer, 2006.
// [2] Wächter, A. and Biegler, L. "On the implementation of an interior-point
//     filter line-search algorithm for large-scale nonlinear programming",
//     2005. http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf
// [3] Byrd, R. and Nocedal J. and Waltz R. "KNITRO: An Integrated Package for
//     Nonlinear Optimization", 2005.
//     https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf

namespace {

/**
 * Applies fraction-to-the-boundary rule to a variable and its iterate, then
 * returns a fraction of the iterate step size within (0, 1].
 *
 * @param x The variable.
 * @param p The iterate on the variable.
 * @param tau Fraction-to-the-boundary rule scaling factor.
 * @return Fraction of the iterate step size within (0, 1].
 */
double FractionToTheBoundaryRule(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& p,
                                 double tau) {
  // αᵐᵃˣ = max(α ∈ (0, 1] : x + αp ≥ (1−τ)x)
  //      = max(α ∈ (0, 1] : αp ≥ −τx)
  double alpha = 1.0;
  for (int i = 0; i < x.rows(); ++i) {
    if (p(i) != 0.0) {
      while (alpha * p(i) < -tau * x(i)) {
        alpha *= 0.999;
      }
    }
  }

  return alpha;
}

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1000.0;
}

}  // namespace

OptimizationProblem::OptimizationProblem() noexcept {
  m_decisionVariables.reserve(1024);
  m_equalityConstraints.reserve(1024);
  m_inequalityConstraints.reserve(1024);
}

Variable OptimizationProblem::DecisionVariable() {
  m_decisionVariables.emplace_back(0.0);
  return m_decisionVariables.back();
}

VariableMatrix OptimizationProblem::DecisionVariable(int rows, int cols) {
  m_decisionVariables.reserve(m_decisionVariables.size() + rows * cols);

  VariableMatrix vars{rows, cols};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      m_decisionVariables.emplace_back(0.0);
      vars(row, col) = m_decisionVariables.back();
    }
  }

  return vars;
}

VariableMatrix OptimizationProblem::SymmetricDecisionVariable(int rows) {
  // An nxn symmetric matrix has (n² + n)/2 unique entries. The number of
  // entries in the lower triangle is equal to the sum of the numbers 1 to n).
  m_decisionVariables.reserve(m_decisionVariables.size() +
                              (rows * rows + rows) / 2);

  VariableMatrix vars{rows, rows};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col <= row; ++col) {
      m_decisionVariables.emplace_back(0.0);
      vars(row, col) = m_decisionVariables.back();
      vars(col, row) = m_decisionVariables.back();
    }
  }

  return vars;
}

void OptimizationProblem::Minimize(const Variable& cost) {
  m_f = cost;
}

void OptimizationProblem::Minimize(Variable&& cost) {
  m_f = std::move(cost);
}

void OptimizationProblem::Maximize(const Variable& cost) {
  // Maximizing an cost function is the same as minimizing its negative
  m_f = -cost;
}

void OptimizationProblem::Maximize(Variable&& cost) {
  // Maximizing an cost function is the same as minimizing its negative
  m_f = -std::move(cost);
}

void OptimizationProblem::SubjectTo(const EqualityConstraints& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(storage[i]);
  }
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(std::move(storage[i]));
  }
}

void OptimizationProblem::SubjectTo(const InequalityConstraints& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(storage[i]);
  }
}

void OptimizationProblem::SubjectTo(InequalityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(std::move(storage[i]));
  }
}

SolverStatus OptimizationProblem::Solve(const SolverConfig& config) {
  m_config = config;

  // Create the initial value column vector
  Eigen::VectorXd x{m_decisionVariables.size(), 1};
  for (size_t i = 0; i < m_decisionVariables.size(); ++i) {
    x(i) = m_decisionVariables[i].Value();
  }

  SolverStatus status;

  // Get f's expression type
  if (m_f.has_value()) {
    status.costFunctionType = m_f.value().Type();
  }

  // Get the highest order equality constraint expression type
  for (const auto& constraint : m_equalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.equalityConstraintType < constraintType) {
      status.equalityConstraintType = constraintType;
    }
  }

  // Get the highest order inequality constraint expression type
  for (const auto& constraint : m_inequalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.inequalityConstraintType < constraintType) {
      status.inequalityConstraintType = constraintType;
    }
  }

  if (m_config.diagnostics) {
    constexpr std::array<const char*, 5> kExprTypeToName = {
        "empty", "constant", "linear", "quadratic", "nonlinear"};

    fmt::print("The cost function is {}.\n",
               kExprTypeToName[static_cast<int>(status.costFunctionType)]);
    fmt::print(
        "The equality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
    fmt::print(
        "The inequality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
    fmt::print("\n");
  }

  // If the problem is empty or constant, there's nothing to do
  if (status.costFunctionType <= ExpressionType::kConstant &&
      status.equalityConstraintType <= ExpressionType::kConstant &&
      status.inequalityConstraintType <= ExpressionType::kConstant) {
    return status;
  }

  // If there's no cost function, make it zero and continue
  if (!m_f.has_value()) {
    m_f = 0.0;
  }

  if (config.spy) {
    m_A_e_spy.open("A_e.spy");
    m_A_i_spy.open("A_i.spy");
    m_H_spy.open("H.spy");
  }

  // Solve the optimization problem
  Eigen::VectorXd solution = InteriorPoint(x, &status);

  if (m_config.diagnostics) {
    fmt::print("Exit condition: ");
    if (status.exitCondition == SolverExitCondition::kOk) {
      fmt::print("optimal solution found");
    } else if (status.exitCondition == SolverExitCondition::kTooFewDOFs) {
      fmt::print("problem has too few degrees of freedom");
    } else if (status.exitCondition ==
               SolverExitCondition::kLocallyInfeasible) {
      fmt::print("problem is locally infeasible");
    } else if (status.exitCondition ==
               SolverExitCondition::kNumericalIssue_BadStep) {
      fmt::print(
          "solver failed to reach the desired tolerance due to a numerical "
          "issue (bad step)");
    } else if (status.exitCondition ==
               SolverExitCondition::kNumericalIssue_MaxStepTooSmall) {
      fmt::print(
          "solver failed to reach the desired tolerance due to a numerical "
          "issue (max step too small)");
    } else if (status.exitCondition == SolverExitCondition::kMaxIterations) {
      fmt::print("maximum iterations exceeded");
    } else if (status.exitCondition == SolverExitCondition::kTimeout) {
      fmt::print("solution returned after timeout");
    }
    fmt::print("\n");
  }

  // Assign the solution to the original Variable instances
  SetAD(m_decisionVariables, solution);

  return status;
}

void OptimizationProblem::Callback(
    std::function<void(const SolverIterationInfo&)> callback) {
  m_callback = callback;
}

Eigen::VectorXd OptimizationProblem::InteriorPoint(
    const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
    SolverStatus* status) {
  // Read docs/algorithms.md#Interior-point_method for a derivation of the
  // interior-point method formulation being used.

  auto solveStartTime = std::chrono::system_clock::now();

  // Print problem dimensionality
  if (m_config.diagnostics) {
    fmt::print("Number of decision variables: {}\n",
               m_decisionVariables.size());
    fmt::print("Number of equality constraints: {}\n",
               m_equalityConstraints.size());
    fmt::print("Number of inequality constraints: {}\n\n",
               m_inequalityConstraints.size());
  }

  // Map decision variables and constraints to Eigen vectors for Lagrangian
  MapVectorXvar xAD(m_decisionVariables.data(), m_decisionVariables.size());
  MapVectorXvar c_eAD(m_equalityConstraints.data(),
                      m_equalityConstraints.size());
  MapVectorXvar c_iAD(m_inequalityConstraints.data(),
                      m_inequalityConstraints.size());

  // Create autodiff variables for s, y, and z for Lagrangian
  VectorXvar sAD = VectorXvar::Ones(m_inequalityConstraints.size());
  VectorXvar yAD = VectorXvar::Zero(m_equalityConstraints.size());
  VectorXvar zAD = VectorXvar::Ones(m_inequalityConstraints.size());

  // Lagrangian L
  //
  // L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  Variable L =
      m_f.value() - yAD.transpose() * c_eAD - zAD.transpose() * (c_iAD - sAD);
  ExpressionGraph graphL{L};

  // Set x to initial guess and update autodiff so Jacobians and Hessian use it
  Eigen::VectorXd x = initialGuess;
  SetAD(xAD, x);

  // Equality constraint Jacobian Aₑ
  //
  //         [∇ᵀcₑ₁(x)ₖ]
  // Aₑ(x) = [∇ᵀcₑ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcₑₘ(x)ₖ]
  Jacobian jacobianCe{c_eAD, xAD};
  Eigen::SparseMatrix<double> A_e = jacobianCe.Calculate();

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(x)ₖ]
  // Aᵢ(x) = [∇ᵀcᵢ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(x)ₖ]
  Jacobian jacobianCi{c_iAD, xAD};
  Eigen::SparseMatrix<double> A_i = jacobianCi.Calculate();

  // Gradient of f ∇f
  Gradient gradientF{m_f.value(), xAD};
  Eigen::SparseVector<double> g = gradientF.Calculate();

  // Initialize y for the Hessian to use
  Eigen::VectorXd y = InitializeY(A_e, g);
  SetAD(yAD, y);

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
  Hessian hessianL{L, xAD};
  Eigen::SparseMatrix<double> H = hessianL.Calculate();

  Eigen::VectorXd s = GetAD(sAD);
  Eigen::VectorXd z = GetAD(zAD);
  Eigen::VectorXd c_e = GetAD(m_equalityConstraints);
  Eigen::VectorXd c_i = GetAD(m_inequalityConstraints);

  // Check for overconstrained problem
  if (m_equalityConstraints.size() > m_decisionVariables.size()) {
    fmt::print("The problem has too few degrees of freedom.\n");
    fmt::print("Violated constraints (cₑ(x) = 0) in order of declaration:\n");
    for (int row = 0; row < c_e.rows(); ++row) {
      if (c_e(row) < 0.0) {
        fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
      }
    }

    status->exitCondition = SolverExitCondition::kTooFewDOFs;
    return x;
  }

  if (m_config.diagnostics) {
    PrintNonZeros(status, H, A_e, A_i);

    fmt::print("Error tolerance: {}\n\n", m_config.tolerance);
  }

  std::chrono::system_clock::time_point iterationsStartTime;

  int iterations = 0;

  scope_exit exit{[&] {
    if (m_config.diagnostics) {
      auto solveEndTime = std::chrono::system_clock::now();

      fmt::print("\nSolve time: {} ms\n",
                 ToMilliseconds(solveEndTime - solveStartTime));
      fmt::print("  ↳ {} ms (IPM setup)\n",
                 ToMilliseconds(iterationsStartTime - solveStartTime));
      if (iterations > 0) {
        fmt::print(
            "  ↳ {} ms ({} IPM iterations; {} ms average)\n",
            ToMilliseconds(solveEndTime - iterationsStartTime), iterations,
            ToMilliseconds((solveEndTime - iterationsStartTime) / iterations));
      }
      fmt::print("\n");

      constexpr auto format = "{:>8}  {:>10}  {:>14}  {:>6}\n";
      fmt::print(format, "autodiff", "setup (ms)", "avg solve (ms)", "solves");
      fmt::print("{:=^44}\n", "");
      fmt::print(format, "∇f(x)", gradientF.GetProfiler().SetupDuration(),
                 gradientF.GetProfiler().AverageSolveDuration(),
                 gradientF.GetProfiler().SolveMeasurements());
      fmt::print(format, "∇²ₓₓL", hessianL.GetProfiler().SetupDuration(),
                 hessianL.GetProfiler().AverageSolveDuration(),
                 hessianL.GetProfiler().SolveMeasurements());
      fmt::print(format, "∂cₑ/∂x", jacobianCe.GetProfiler().SetupDuration(),
                 jacobianCe.GetProfiler().AverageSolveDuration(),
                 jacobianCe.GetProfiler().SolveMeasurements());
      fmt::print(format, "∂cᵢ/∂x", jacobianCi.GetProfiler().SetupDuration(),
                 jacobianCi.GetProfiler().AverageSolveDuration(),
                 jacobianCi.GetProfiler().SolveMeasurements());
      fmt::print("\n");
    }
  }};

  // Barrier parameter minimum
  double mu_min = m_config.tolerance / 10.0;

  // Barrier parameter μ
  double mu = 0.1;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double tau_min = 0.99;

  // Fraction-to-the-boundary rule scale factor τ
  double tau = tau_min;

  Filter filter{FilterEntry{m_f.value(), mu, s, c_e, c_i}};

  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter
  auto UpdateBarrierParameterAndResetFilter = [&] {
    // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
    constexpr double kappa_mu = 0.2;

    // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1,
    // 2).
    constexpr double theta_mu = 1.5;

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) of [2].
    mu = std::max(mu_min, std::min(kappa_mu * mu, std::pow(mu, theta_mu)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    tau = std::max(tau_min, 1.0 - mu);

    // Reset the filter when the barrier parameter is updated
    filter.Reset(FilterEntry{m_f.value(), mu, s, c_e, c_i});
  };

  // Kept outside the loop so its storage can be reused
  std::vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver;

  int stepTooSmallCounter = 0;

  // Error estimate E_μ
  double E_mu = std::numeric_limits<double>::infinity();

  iterationsStartTime = std::chrono::system_clock::now();

  while (E_mu > m_config.tolerance) {
    // Update autodiff for Jacobians and Hessian
    SetAD(xAD, x);
    SetAD(sAD, s);
    SetAD(yAD, y);
    SetAD(zAD, z);

    auto innerIterStartTime = std::chrono::system_clock::now();

    //     [s₁ 0 ⋯ 0 ]
    // S = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 sₘ]
    Eigen::SparseMatrix<double> S = SparseDiagonal(s);

    //         [∇ᵀcₑ₁(x)ₖ]
    // Aₑ(x) = [∇ᵀcₑ₂(x)ₖ]
    //         [    ⋮    ]
    //         [∇ᵀcₑₘ(x)ₖ]
    A_e = jacobianCe.Calculate();

    //         [∇ᵀcᵢ₁(x)ₖ]
    // Aᵢ(x) = [∇ᵀcᵢ₂(x)ₖ]
    //         [    ⋮    ]
    //         [∇ᵀcᵢₘ(x)ₖ]
    A_i = jacobianCi.Calculate();

    // Update cₑ and cᵢ
    c_e = GetAD(m_equalityConstraints);
    c_i = GetAD(m_inequalityConstraints);

    // Check for local infeasibility
    if (!IsLocallyFeasible(A_e, c_e, A_i, c_i)) {
      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return x;
    }

    // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
    H = hessianL.Calculate();

    g = gradientF.Calculate();

    if (m_config.spy) {
      // Gap between sparsity patterns
      if (iterations > 0) {
        m_A_e_spy << "\n";
        m_A_i_spy << "\n";
        m_H_spy << "\n";
      }

      Spy(m_H_spy, H);
      Spy(m_A_e_spy, A_e);
      Spy(m_A_i_spy, A_i);
    }

    // Call user callback
    m_callback({iterations, g, H, A_e, A_i});

    // If the error estimate is below the desired threshold for this barrier
    // parameter value, decrease it further and restart the loop
    {
      // Barrier parameter scale factor κ_μ for tolerance checks
      constexpr double kappa_epsilon = 10.0;

      E_mu = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, S, y, z, mu);
      if (E_mu <= kappa_epsilon * mu) {
        UpdateBarrierParameterAndResetFilter();
        continue;
      }
    }

    //     [z₁ 0 ⋯ 0 ]
    // Z = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 zₘ]
    Eigen::SparseMatrix<double> Z = SparseDiagonal(z);

    // Σ = S⁻¹Z
    Eigen::SparseMatrix<double> sigma = S.cwiseInverse() * Z;

    // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
    //       [    Aₑ       0 ]
    triplets.clear();
    // Assign top-left quadrant
    AssignSparseBlock(triplets, 0, 0, H + A_i.transpose() * sigma * A_i);
    // Assign bottom-left quadrant
    AssignSparseBlock(triplets, H.rows(), 0, A_e);
    // Assign top-right quadrant
    AssignSparseBlock(triplets, 0, H.rows(), A_e.transpose());
    Eigen::SparseMatrix<double> lhs{H.rows() + A_e.rows(),
                                    H.cols() + A_e.rows()};
    lhs.setFromTriplets(triplets.begin(), triplets.end());

    const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

    // rhs = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
    //        [                cₑ                ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) =
        -(g - A_e.transpose() * y +
          A_i.transpose() * (S.cwiseInverse() * (Z * c_i - mu * e) - z));
    rhs.segment(x.rows(), y.rows()) = -c_e;

    // Solve the Newton-KKT system
    solver.Compute(lhs, m_equalityConstraints.size(), mu);
    Eigen::VectorXd step{x.rows() + y.rows(), 1};
    if (solver.Info() == Eigen::Success) {
      step = solver.Solve(rhs);
    } else {
      // The regularization procedure failed due to a rank-deficient equality
      // constraint Jacobian with linearly dependent constraints. Set the step
      // length to zero and let second-order corrections attempt to restore
      // feasibility.
      step.setZero();
    }

    // step = [ pₖˣ]
    //        [−pₖʸ]
    Eigen::VectorXd p_x = step.segment(0, x.rows());
    Eigen::VectorXd p_y = -step.segment(x.rows(), y.rows());

    // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
    Eigen::VectorXd p_z =
        -sigma * c_i + mu * S.cwiseInverse() * e - sigma * A_i * p_x;

    // pₖˢ = μZ⁻¹e − s − Z⁻¹Spₖᶻ
    Eigen::VectorXd p_s =
        mu * Z.cwiseInverse() * e - s - Z.cwiseInverse() * S * p_z;

    bool stepAcceptable = false;

    double alpha_max = FractionToTheBoundaryRule(s, p_s, tau);
    double alpha = alpha_max;

    while (!stepAcceptable) {
      Eigen::VectorXd trial_x = x + alpha * p_x;
      Eigen::VectorXd trial_s = s + alpha * p_s;

      double alpha_z = FractionToTheBoundaryRule(z, p_z, tau);
      Eigen::VectorXd trial_y = y + alpha_z * p_y;
      Eigen::VectorXd trial_z = z + alpha_z * p_z;

      SetAD(xAD, trial_x);
      m_f.value().Update();

      for (int row = 0; row < c_e.rows(); ++row) {
        c_eAD(row).Update();
      }
      Eigen::VectorXd trial_c_e = GetAD(m_equalityConstraints);

      for (int row = 0; row < c_i.rows(); ++row) {
        c_iAD(row).Update();
      }
      Eigen::VectorXd trial_c_i = GetAD(m_inequalityConstraints);

      FilterEntry entry{m_f.value(), mu, trial_s, trial_c_e, trial_c_i};
      if (filter.IsAcceptable(entry)) {
        stepAcceptable = true;
        filter.Add(std::move(entry));
        continue;
      }

      double prevConstraintViolation = c_e.lpNorm<1>() + (c_i - s).lpNorm<1>();
      double nextConstraintViolation =
          trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();

      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (nextConstraintViolation >= prevConstraintViolation) {
        // Apply second-order corrections. See section 2.4 of [2].
        Eigen::VectorXd p_x_cor = p_x;
        Eigen::VectorXd p_y_soc = p_y;
        Eigen::VectorXd p_z_soc = p_z;
        Eigen::VectorXd p_s_soc = p_s;

        double alpha_soc = alpha;
        Eigen::VectorXd c_e_soc = c_e;

        for (int soc_iteration = 0; soc_iteration < 5 && !stepAcceptable;
             ++soc_iteration) {
          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
          //        [              cₑˢᵒᶜ               ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αp_x)
          c_e_soc = alpha_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          step = solver.Solve(rhs);

          p_x_cor = step.segment(0, x.rows());
          p_y_soc = -step.segment(x.rows(), y.rows());

          // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
          p_z_soc =
              -sigma * c_i + mu * S.cwiseInverse() * e - sigma * A_i * p_x_cor;

          // pₖˢ = μZ⁻¹e − s − Z⁻¹Spₖᶻ
          p_s_soc =
              mu * Z.cwiseInverse() * e - s - Z.cwiseInverse() * S * p_z_soc;

          alpha_soc = FractionToTheBoundaryRule(s, p_s_soc, tau);
          trial_x = x + alpha_soc * p_x_cor;
          trial_s = s + alpha_soc * p_s_soc;

          // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
          alpha_z = FractionToTheBoundaryRule(z, p_z_soc, tau);
          trial_y = y + alpha_z * p_y_soc;
          trial_z = z + alpha_z * p_z_soc;

          SetAD(xAD, trial_x);
          m_f.value().Update();

          for (int row = 0; row < c_e.rows(); ++row) {
            c_eAD(row).Update();
          }
          trial_c_e = GetAD(m_equalityConstraints);

          for (int row = 0; row < c_i.rows(); ++row) {
            c_iAD(row).Update();
          }
          trial_c_i = GetAD(m_inequalityConstraints);

          entry = FilterEntry{m_f.value(), mu, trial_s, trial_c_e, trial_c_i};
          if (filter.IsAcceptable(entry)) {
            p_x = p_x_cor;
            p_y = p_y_soc;
            p_z = p_z_soc;
            p_s = p_s_soc;
            alpha = alpha_soc;
            stepAcceptable = true;
            filter.Add(std::move(entry));
          }
        }
      }

      if (!stepAcceptable) {
        alpha *= 0.5;

        // Safety factor for the minimal step size
        constexpr double alpha_min_frac = 0.05;

        if (alpha < alpha_min_frac * 1e-5) {
          if (mu > mu_min) {
            UpdateBarrierParameterAndResetFilter();
            break;
          } else {
            status->exitCondition =
                SolverExitCondition::kNumericalIssue_BadStep;
            return x;
          }
        }
      }
    }

    if (stepAcceptable) {
      // Handle very small search directions by letting αₖ = αₖᵐᵃˣ when
      // max(|pₖˣ(i)|/(1 + |xₖ(i)|)) < 10ε_mach.
      //
      // See section 3.9 of [2].
      double maxStepScaled = 0.0;
      for (int row = 0; row < x.rows(); ++row) {
        maxStepScaled = std::max(maxStepScaled,
                                 std::abs(p_x(row)) / (1.0 + std::abs(x(row))));
      }
      if (maxStepScaled < 10.0 * std::numeric_limits<double>::epsilon()) {
        alpha = alpha_max;
        ++stepTooSmallCounter;
      } else {
        stepTooSmallCounter = 0;
      }

      // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
      double alpha_z = FractionToTheBoundaryRule(z, p_z, tau);

      // xₖ₊₁ = xₖ + αₖpₖˣ
      // sₖ₊₁ = xₖ + αₖpₖˢ
      // yₖ₊₁ = xₖ + αₖᶻpₖʸ
      // zₖ₊₁ = xₖ + αₖᶻpₖᶻ
      x += alpha * p_x;
      s += alpha * p_s;
      y += alpha_z * p_y;
      z += alpha_z * p_z;

      // A requirement for the convergence proof is that the "primal-dual
      // barrier term Hessian" Σₖ does not deviate arbitrarily much from the
      // "primal Hessian" μⱼSₖ⁻². We ensure this by resetting
      //
      //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
      //
      // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
      {
        // Barrier parameter scale factor κ_Σ for inequality constraint Lagrange
        // multiplier safeguard
        constexpr double kappa_sigma = 1e10;

        for (int row = 0; row < z.rows(); ++row) {
          z(row) = std::max(std::min(z(row), kappa_sigma * mu / s(row)),
                            mu / (kappa_sigma * s(row)));
        }
      }
    }

    auto innerIterEndTime = std::chrono::system_clock::now();

    if (m_config.diagnostics) {
      if (iterations % 20 == 0) {
        fmt::print("{:>4}   {:>10}  {:>10}   {:>16}  {:>19}\n", "iter",
                   "time (ms)", "error", "cost", "infeasibility");
        fmt::print("{:=^70}\n", "");
      }
      fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations,
                 ToMilliseconds(innerIterEndTime - innerIterStartTime), E_mu,
                 filter.LastEntry().cost,
                 filter.LastEntry().constraintViolation);
    }

    ++iterations;
    if (iterations >= m_config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterations;
      return x;
    }

    if (innerIterEndTime - solveStartTime > m_config.timeout) {
      status->exitCondition = SolverExitCondition::kTimeout;
      return x;
    }

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (stepTooSmallCounter == 2) {
      if (mu > mu_min) {
        UpdateBarrierParameterAndResetFilter();
        continue;
      } else {
        status->exitCondition =
            SolverExitCondition::kNumericalIssue_MaxStepTooSmall;
        return x;
      }
    }
  }

  if (m_config.diagnostics) {
    fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations, 0.0,
               E_mu, m_f.value().Value() - mu * s.array().log().sum(),
               c_e.lpNorm<1>() + (c_i - s).lpNorm<1>());
  }

  return x;
}

Eigen::VectorXd OptimizationProblem::InitializeY(
    const Eigen::SparseMatrix<double>& A_e, const Eigen::VectorXd& g) const {
  //   [  I     Aₑᵀ(x₀)][w] = −[∇f(x₀)]
  //   [Aₑ(x₀)     0   ][y]    [  0   ]
  //
  // See equation (36) of [2].

  std::vector<Eigen::Triplet<double>> triplets;

  // Assign top-left quadrant
  AssignSparseBlock(
      triplets, 0, 0,
      SparseIdentity(m_decisionVariables.size(), m_decisionVariables.size()));
  // Assign bottom-left quadrant
  AssignSparseBlock(triplets, m_decisionVariables.size(), 0, A_e);
  // Assign top-right quadrant
  AssignSparseBlock(triplets, 0, m_decisionVariables.size(), A_e.transpose());

  // [  I     Aₑᵀ(x₀)]
  // [Aₑ(x₀)     0   ]
  Eigen::SparseMatrix<double> lhs(m_decisionVariables.size() + A_e.rows(),
                                  m_decisionVariables.size() + A_e.rows());
  lhs.setFromTriplets(triplets.begin(), triplets.end());

  // −[∇f(x₀)]
  //  [  0   ]
  Eigen::VectorXd rhs(m_decisionVariables.size() + A_e.rows());
  rhs.block(0, 0, m_decisionVariables.size(), 1) = -g;
  rhs.block(m_decisionVariables.size(), 0, A_e.rows(), 1).setZero();

  Eigen::VectorXd y =
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>{lhs}.solve(rhs).block(
          m_decisionVariables.size(), 0, A_e.rows(), 1);
  if (y.lpNorm<Eigen::Infinity>() > 1e3) {
    return Eigen::VectorXd::Zero(A_e.rows());
  } else {
    return y;
  }
}

void OptimizationProblem::PrintNonZeros(
    SolverStatus* status, const Eigen::SparseMatrix<double>& H,
    const Eigen::SparseMatrix<double>& A_e,
    const Eigen::SparseMatrix<double>& A_i) {
  std::string prints;

  if (status->costFunctionType <= ExpressionType::kQuadratic &&
      status->equalityConstraintType <= ExpressionType::kQuadratic &&
      status->inequalityConstraintType <= ExpressionType::kQuadratic) {
    prints += fmt::format("Number of nonzeros in Lagrangian Hessian: {}\n",
                          H.nonZeros());
  }
  if (status->equalityConstraintType <= ExpressionType::kLinear) {
    prints +=
        fmt::format("Number of nonzeros in equality constraint Jacobian: {}\n",
                    A_e.nonZeros());
  }
  if (status->inequalityConstraintType <= ExpressionType::kLinear) {
    prints += fmt::format(
        "Number of nonzeros in inequality constraint Jacobian: {}\n",
        A_i.nonZeros());
  }

  if (prints.length() > 0) {
    fmt::print("{}\n", prints);
  }
}

bool OptimizationProblem::IsLocallyFeasible(
    const Eigen::SparseMatrix<double>& A_e, const Eigen::VectorXd& c_e,
    const Eigen::SparseMatrix<double>& A_i, const Eigen::VectorXd& c_i) const {
  // Check for problem local infeasibility. The problem is locally infeasible if
  //
  //   Aₑᵀcₑ → 0
  //   Aᵢᵀcᵢ⁺ → 0
  //   ‖(cₑ, cᵢ⁺)‖ > ε
  //
  // where cᵢ⁺ = min(cᵢ, 0).
  //
  // See "Infeasibility detection" in section 6 of [3].
  //
  // cᵢ⁺ is used instead of cᵢ⁻ from the paper to follow the convention that
  // feasible inequality constraints are ≥ 0.

  if (m_equalityConstraints.size() > 0 &&
      (A_e.transpose() * c_e).norm() < 1e-6 && c_e.norm() > 1e-2) {
    if (m_config.diagnostics) {
      fmt::print(
          "The problem is locally infeasible due to violated equality "
          "constraints.\n");
      fmt::print("Violated constraints (cₑ(x) = 0) in order of declaration:\n");
      for (int row = 0; row < c_e.rows(); ++row) {
        if (c_e(row) < 0.0) {
          fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
        }
      }
    }

    return false;
  }

  if (m_inequalityConstraints.size() > 0) {
    Eigen::VectorXd c_i_plus = c_i.cwiseMin(0.0);
    if ((A_i.transpose() * c_i_plus).norm() < 1e-6 && c_i_plus.norm() > 1e-6) {
      if (m_config.diagnostics) {
        fmt::print(
            "The problem is infeasible due to violated inequality "
            "constraints.\n");
        fmt::print(
            "Violated constraints (cᵢ(x) ≥ 0) in order of declaration:\n");
        for (int row = 0; row < c_i.rows(); ++row) {
          if (c_i(row) < 0.0) {
            fmt::print("  {}/{}: {} ≥ 0\n", row + 1, c_i.rows(), c_i(row));
          }
        }
      }

      return false;
    }
  }

  return true;
}

double OptimizationProblem::ErrorEstimate(
    const Eigen::VectorXd& g, const Eigen::SparseMatrix<double>& A_e,
    const Eigen::VectorXd& c_e, const Eigen::SparseMatrix<double>& A_i,
    const Eigen::VectorXd& c_i, const Eigen::VectorXd& s,
    const Eigen::SparseMatrix<double>& S, const Eigen::VectorXd& y,
    const Eigen::VectorXd& z, double mu) const {
  // Update the error estimate using the KKT conditions from equations (19.5a)
  // through (19.5d) of [1].
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // The error tolerance is the max of the following infinity norms scaled by
  // s_d and s_c (see equation (5) of [2]).
  //
  //   ‖∇f − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
  //   ‖Sz − μe‖_∞ / s_c
  //   ‖cₑ‖_∞
  //   ‖cᵢ − s‖_∞

  // s_d = max(sₘₐₓ, (‖y‖₁ + ‖z‖₁) / (m + n)) / sₘₐₓ
  constexpr double s_max = 100.0;
  double s_d = std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) /
                                   (m_equalityConstraints.size() +
                                    m_inequalityConstraints.size())) /
               s_max;

  // s_c = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
  double s_c =
      std::max(s_max, z.lpNorm<1>() / m_inequalityConstraints.size()) / s_max;

  const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  double E_mu = std::max((g - A_e.transpose() * y - A_i.transpose() * z)
                                 .lpNorm<Eigen::Infinity>() /
                             s_d,
                         (S * z - mu * e).lpNorm<Eigen::Infinity>() / s_c);
  if (m_equalityConstraints.size() > 0) {
    E_mu = std::max(E_mu, c_e.lpNorm<Eigen::Infinity>());
  }
  if (m_inequalityConstraints.size() > 0) {
    E_mu = std::max(E_mu, (c_i - s).lpNorm<Eigen::Infinity>());
  }

  return E_mu;
}
