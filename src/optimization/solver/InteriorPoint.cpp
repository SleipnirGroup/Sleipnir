// Copyright (c) Sleipnir contributors

#include "InteriorPoint.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <limits>

#include <fmt/core.h>

#include "optimization/Filter.hpp"
#include "optimization/RegularizedLDLT.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/AutodiffUtil.hpp"
#include "util/ScopeExit.hpp"
#include "util/SparseMatrixBuilder.hpp"

namespace sleipnir {

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
 * Returns true if the problem's equality constraints are locally infeasible.
 *
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 */
bool IsEqualityLocallyInfeasible(const Eigen::SparseMatrix<double>& A_e,
                                 const Eigen::VectorXd& c_e) {
  // The equality constraints are locally infeasible if
  //
  //   Aₑᵀcₑ → 0
  //   ‖cₑ‖ > ε
  //
  // See "Infeasibility detection" in section 6 of [3].
  return A_e.rows() > 0 && (A_e.transpose() * c_e).norm() < 1e-6 &&
         c_e.norm() > 1e-2;
}

/**
 * Returns true if the problem's inequality constraints are locally infeasible.
 *
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 */
bool IsInequalityLocallyInfeasible(const Eigen::SparseMatrix<double>& A_i,
                                   const Eigen::VectorXd& c_i) {
  // The inequality constraints are locally infeasible if
  //
  //   Aᵢᵀcᵢ⁺ → 0
  //   ‖cᵢ⁺‖ > ε
  //
  // where cᵢ⁺ = min(cᵢ, 0).
  //
  // See "Infeasibility detection" in section 6 of [3].
  //
  // cᵢ⁺ is used instead of cᵢ⁻ from the paper to follow the convention that
  // feasible inequality constraints are ≥ 0.
  if (A_i.rows() > 0) {
    Eigen::VectorXd c_i_plus = c_i.cwiseMin(0.0);
    if ((A_i.transpose() * c_i_plus).norm() < 1e-6 && c_i_plus.norm() > 1e-6) {
      return true;
    }
  }

  return false;
}

/**
 * Returns the error estimate using the KKT conditions for the interior-point
 * method.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 * @param s Inequality constraint slack variables.
 * @param y Equality constraint dual variables.
 * @param z Inequality constraint dual variables.
 * @param μ Barrier parameter.
 */
double ErrorEstimate(const Eigen::VectorXd& g,
                     const Eigen::SparseMatrix<double>& A_e,
                     const Eigen::VectorXd& c_e,
                     const Eigen::SparseMatrix<double>& A_i,
                     const Eigen::VectorXd& c_i, const Eigen::VectorXd& s,
                     const Eigen::VectorXd& y, const Eigen::VectorXd& z,
                     double μ) {
  int numEqualityConstraints = A_e.rows();
  int numInequalityConstraints = A_i.rows();

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
  double s_d =
      std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) /
                          (numEqualityConstraints + numInequalityConstraints)) /
      s_max;

  // s_c = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
  double s_c =
      std::max(s_max, z.lpNorm<1>() / numInequalityConstraints) / s_max;

  const auto S = s.asDiagonal();
  const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  return std::max({(g - A_e.transpose() * y - A_i.transpose() * z)
                           .lpNorm<Eigen::Infinity>() /
                       s_d,
                   (S * z - μ * e).lpNorm<Eigen::Infinity>() / s_c,
                   c_e.lpNorm<Eigen::Infinity>(),
                   (c_i - s).lpNorm<Eigen::Infinity>()});
}

/**
 * Returns the KKT error for the interior-point method.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 * @param s Inequality constraint slack variables.
 * @param y Equality constraint dual variables.
 * @param z Inequality constraint dual variables.
 * @param μ Barrier parameter.
 */
double KKTError(const Eigen::VectorXd& g,
                const Eigen::SparseMatrix<double>& A_e,
                const Eigen::VectorXd& c_e,
                const Eigen::SparseMatrix<double>& A_i,
                const Eigen::VectorXd& c_i, const Eigen::VectorXd& s,
                const Eigen::VectorXd& y, const Eigen::VectorXd& z, double μ) {
  // Compute the KKT error as the 1-norm of the KKT conditions from equations
  // (19.5a) through (19.5d) of [1].
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0

  const auto S = s.asDiagonal();
  const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  return (g - A_e.transpose() * y - A_i.transpose() * z).lpNorm<1>() +
         (S * z - μ * e).lpNorm<1>() + c_e.lpNorm<1>() + (c_i - s).lpNorm<1>();
}

/**
 * Applies fraction-to-the-boundary rule to a variable and its iterate, then
 * returns a fraction of the iterate step size within (0, 1].
 *
 * @param x The variable.
 * @param p The iterate on the variable.
 * @param τ Fraction-to-the-boundary rule scaling factor within (0, 1].
 * @return Fraction of the iterate step size within (0, 1].
 */
double FractionToTheBoundaryRule(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& p,
                                 double τ) {
  // α = max(α ∈ (0, 1] : x + αp ≥ (1 − τ)x)
  //
  // where x and τ are positive.
  //
  // x + αp ≥ (1 − τ)x
  // x + αp ≥ x − τx
  // αp ≥ −τx
  //
  // If the inequality is false, p < 0 and α is too big. Find the largest value
  // of α that makes the inequality true.
  //
  // α = −τ/p x
  double α = 1.0;
  for (int i = 0; i < x.rows(); ++i) {
    if (α * p(i) < -τ * x(i)) {
      α = -τ / p(i) * x(i);
    }
  }

  return α;
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

Eigen::VectorXd InteriorPoint(
    std::vector<Variable>& decisionVariables, std::optional<Variable>& f,
    std::vector<Variable>& equalityConstraints,
    std::vector<Variable>& inequalityConstraints,
    const std::function<bool(const SolverIterationInfo&)>& callback,
    const SolverConfig& config,
    const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
    SolverStatus* status) {
  // Read docs/algorithms.md#Interior-point_method for a derivation of the
  // interior-point method formulation being used.

  auto solveStartTime = std::chrono::system_clock::now();

  // Map decision variables and constraints to Eigen vectors for Lagrangian
  MapVectorXvar xAD(decisionVariables.data(), decisionVariables.size());
  SetAD(xAD, initialGuess);
  MapVectorXvar c_eAD(equalityConstraints.data(), equalityConstraints.size());
  MapVectorXvar c_iAD(inequalityConstraints.data(),
                      inequalityConstraints.size());

  // Create autodiff variables for s, y, and z for Lagrangian
  VectorXvar sAD{inequalityConstraints.size()};
  for (auto& s : sAD) {
    s.SetValue(1.0);
  }
  VectorXvar yAD{equalityConstraints.size()};
  for (auto& y : yAD) {
    y.SetValue(0.0);
  }
  VectorXvar zAD{inequalityConstraints.size()};
  for (auto& z : zAD) {
    z.SetValue(1.0);
  }

  // Lagrangian L
  //
  // L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  Variable L =
      f.value() - yAD.transpose() * c_eAD - zAD.transpose() * (c_iAD - sAD);

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
  Gradient gradientF{f.value(), xAD};
  Eigen::SparseVector<double> g = gradientF.Calculate();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
  Hessian hessianL{L, xAD};
  Eigen::SparseMatrix<double> H = hessianL.Calculate();

  Eigen::VectorXd x = initialGuess;
  Eigen::VectorXd s = GetAD(sAD);
  Eigen::VectorXd y = GetAD(yAD);
  Eigen::VectorXd z = GetAD(zAD);
  Eigen::VectorXd c_e = GetAD(c_eAD);
  Eigen::VectorXd c_i = GetAD(c_iAD);

  // Check for overconstrained problem
  if (equalityConstraints.size() > decisionVariables.size()) {
    if (config.diagnostics) {
      fmt::print("The problem has too few degrees of freedom.\n");
      fmt::print("Violated constraints (cₑ(x) = 0) in order of declaration:\n");
      for (int row = 0; row < c_e.rows(); ++row) {
        if (c_e(row) < 0.0) {
          fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
        }
      }
    }

    status->exitCondition = SolverExitCondition::kTooFewDOFs;
    return x;
  }

  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::ofstream H_spy;
  std::ofstream A_e_spy;
  std::ofstream A_i_spy;
  if (config.spy) {
    A_e_spy.open("A_e.spy");
    A_i_spy.open("A_i.spy");
    H_spy.open("H.spy");
  }

  if (config.diagnostics) {
    fmt::print("Error tolerance: {}\n\n", config.tolerance);
  }

  std::chrono::system_clock::time_point iterationsStartTime;

  int iterations = 0;

  scope_exit exit{[&] {
    if (config.diagnostics) {
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
  double μ_min = config.tolerance / 10.0;

  // Barrier parameter μ
  double μ = 0.1;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double τ_min = 0.99;

  // Fraction-to-the-boundary rule scale factor τ
  double τ = τ_min;

  Filter filter;

  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter
  auto UpdateBarrierParameterAndResetFilter = [&] {
    // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
    constexpr double κ_μ = 0.2;

    // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1,
    // 2).
    constexpr double θ_μ = 1.5;

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) of [2].
    μ = std::max(μ_min, std::min(κ_μ * μ, std::pow(μ, θ_μ)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    τ = std::max(τ_min, 1.0 - μ);

    // Reset the filter when the barrier parameter is updated
    filter.Reset();
  };

  // Kept outside the loop so its storage can be reused
  SparseMatrixBuilder<double> lhsBuilder(
      decisionVariables.size() + equalityConstraints.size(),
      decisionVariables.size() + equalityConstraints.size());

  RegularizedLDLT solver;

  int acceptableIterCounter = 0;
  constexpr int maxAcceptableIterations = 15;
  const double acceptableTolerance = config.tolerance * 100;

  int fullStepRejectedCounter = 0;
  int stepTooSmallCounter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  iterationsStartTime = std::chrono::system_clock::now();

  while (E_0 > config.tolerance &&
         acceptableIterCounter < maxAcceptableIterations) {
    auto innerIterStartTime = std::chrono::system_clock::now();

    // Check for local infeasibility
    if (IsEqualityLocallyInfeasible(A_e, c_e)) {
      if (config.diagnostics) {
        fmt::print(
            "The problem is locally infeasible due to violated equality "
            "constraints.\n");
        fmt::print(
            "Violated constraints (cₑ(x) = 0) in order of declaration:\n");
        for (int row = 0; row < c_e.rows(); ++row) {
          if (c_e(row) < 0.0) {
            fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
          }
        }
      }

      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return x;
    }
    if (IsInequalityLocallyInfeasible(A_i, c_i)) {
      if (config.diagnostics) {
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

      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return x;
    }

    if (config.spy) {
      // Gap between sparsity patterns
      if (iterations > 0) {
        A_e_spy << "\n";
        A_i_spy << "\n";
        H_spy << "\n";
      }

      Spy(H_spy, H);
      Spy(A_e_spy, A_e);
      Spy(A_i_spy, A_i);
    }

    // Call user callback
    if (callback({iterations, x, g, H, A_e, A_i})) {
      status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
      return x;
    }

    //     [s₁ 0 ⋯ 0 ]
    // S = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 sₘ]
    const auto S = s.asDiagonal();
    Eigen::SparseMatrix<double> Sinv;
    Sinv = s.cwiseInverse().asDiagonal();

    //     [z₁ 0 ⋯ 0 ]
    // Z = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 zₘ]
    const auto Z = z.asDiagonal();
    Eigen::SparseMatrix<double> Zinv;
    Zinv = z.cwiseInverse().asDiagonal();

    // Σ = S⁻¹Z
    const Eigen::SparseMatrix<double> Σ = Sinv * Z;

    // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
    //       [    Aₑ       0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    lhsBuilder.Clear();
    // Assign top-left quadrant
    lhsBuilder.Block(0, 0, H.rows(), H.cols()) =
        H.triangularView<Eigen::Lower>() +
        (A_i.transpose() * Σ * A_i).triangularView<Eigen::Lower>();
    // Assign bottom-left quadrant
    lhsBuilder.Block(H.rows(), 0, A_e.rows(), A_e.cols()) = A_e;
    Eigen::SparseMatrix<double> lhs = lhsBuilder.Build();

    const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

    // rhs = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
    //        [                cₑ                ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) =
        -(g - A_e.transpose() * y +
          A_i.transpose() * (Sinv * (Z * c_i - μ * e) - z));
    rhs.segment(x.rows(), y.rows()) = -c_e;

    // Solve the Newton-KKT system
    solver.Compute(lhs, equalityConstraints.size(), μ);
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
    Eigen::VectorXd p_z = -Σ * c_i + μ * Sinv * e - Σ * A_i * p_x;

    // pₖˢ = μZ⁻¹e − s − Z⁻¹Spₖᶻ
    Eigen::VectorXd p_s = μ * Zinv * e - s - Zinv * S * p_z;

    bool stepAcceptable = false;

    // αᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
    double α_max = FractionToTheBoundaryRule(s, p_s, τ);
    double α = α_max;

    // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
    double α_z = FractionToTheBoundaryRule(z, p_z, τ);

    while (!stepAcceptable) {
      Eigen::VectorXd trial_x = x + α * p_x;
      Eigen::VectorXd trial_s = s + α * p_s;

      Eigen::VectorXd trial_y = y + α_z * p_y;
      Eigen::VectorXd trial_z = z + α_z * p_z;

      SetAD(xAD, trial_x);

      for (int row = 0; row < c_e.rows(); ++row) {
        c_eAD(row).Update();
      }
      Eigen::VectorXd trial_c_e = GetAD(c_eAD);

      for (int row = 0; row < c_i.rows(); ++row) {
        c_iAD(row).Update();
      }
      Eigen::VectorXd trial_c_i = GetAD(c_iAD);

      f.value().Update();
      FilterEntry entry{f.value(), μ, trial_s, trial_c_e, trial_c_i};
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

        double α_soc = α;
        Eigen::VectorXd c_e_soc = c_e;

        for (int soc_iteration = 0; soc_iteration < 5 && !stepAcceptable;
             ++soc_iteration) {
          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
          //        [              cₑˢᵒᶜ               ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αp_x)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          step = solver.Solve(rhs);

          p_x_cor = step.segment(0, x.rows());
          p_y_soc = -step.segment(x.rows(), y.rows());

          // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
          p_z_soc = -Σ * c_i + μ * Sinv * e - Σ * A_i * p_x_cor;

          // pₖˢ = μZ⁻¹e − s − Z⁻¹Spₖᶻ
          p_s_soc = μ * Zinv * e - s - Zinv * S * p_z_soc;

          // αˢᵒᶜ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
          α_soc = FractionToTheBoundaryRule(s, p_s_soc, τ);
          trial_x = x + α_soc * p_x_cor;
          trial_s = s + α_soc * p_s_soc;

          // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
          double α_z_soc = FractionToTheBoundaryRule(z, p_z_soc, τ);
          trial_y = y + α_z_soc * p_y_soc;
          trial_z = z + α_z_soc * p_z_soc;

          SetAD(xAD, trial_x);

          for (int row = 0; row < c_e.rows(); ++row) {
            c_eAD(row).Update();
          }
          trial_c_e = GetAD(c_eAD);

          for (int row = 0; row < c_i.rows(); ++row) {
            c_iAD(row).Update();
          }
          trial_c_i = GetAD(c_iAD);

          f.value().Update();
          entry = FilterEntry{f.value(), μ, trial_s, trial_c_e, trial_c_i};
          if (filter.IsAcceptable(entry)) {
            p_x = p_x_cor;
            p_y = p_y_soc;
            p_z = p_z_soc;
            p_s = p_s_soc;
            α = α_soc;
            α_z = α_z_soc;
            stepAcceptable = true;
            filter.Add(std::move(entry));
          }
        }
      }

      // Count number of times full step is rejected in a row
      if (α == 1.0) {
        if (!stepAcceptable) {
          ++fullStepRejectedCounter;
        } else {
          fullStepRejectedCounter = 0;
        }
      }

      if (!stepAcceptable) {
        // Reset filter if it's repeatedly impeding progress
        //
        // See section 3.2 case I of [2].
        if (fullStepRejectedCounter == 4 &&
            filter.maxConstraintViolation > entry.constraintViolation / 10.0) {
          filter.maxConstraintViolation *= 0.1;
          filter.Reset();
          continue;
        }

        constexpr double α_red_factor = 0.5;
        α *= α_red_factor;

        // Safety factor for the minimal step size
        constexpr double α_min_frac = 0.05;

        if (α < α_min_frac * Filter::γConstraint) {
          // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway

          double currentKKTError = KKTError(g, A_e, c_e, A_i, c_i, s, y, z, μ);

          Eigen::VectorXd trial_x = x + α_max * p_x;
          Eigen::VectorXd trial_s = s + α_max * p_s;

          Eigen::VectorXd trial_y = y + α_z * p_y;
          Eigen::VectorXd trial_z = z + α_z * p_z;

          // Upate autodiff
          SetAD(xAD, trial_x);
          SetAD(sAD, trial_s);
          SetAD(yAD, trial_y);
          SetAD(zAD, trial_z);

          for (int row = 0; row < c_e.rows(); ++row) {
            c_eAD(row).Update();
          }
          Eigen::VectorXd trial_c_e = GetAD(c_eAD);

          for (int row = 0; row < c_i.rows(); ++row) {
            c_iAD(row).Update();
          }
          Eigen::VectorXd trial_c_i = GetAD(c_iAD);

          double nextKKTError = KKTError(
              gradientF.Calculate(), jacobianCe.Calculate(), trial_c_e,
              jacobianCi.Calculate(), trial_c_i, trial_s, trial_y, trial_z, μ);

          if (nextKKTError <= 0.999 * currentKKTError) {
            α = α_max;
            stepAcceptable = true;
            continue;
          }

          // TODO: Feasibility restoration phase
          status->exitCondition = SolverExitCondition::kBadSearchDirection;
          return x;
        }
      }
    }

    if (p_x.lpNorm<Eigen::Infinity>() > 1e20 ||
        p_s.lpNorm<Eigen::Infinity>() > 1e20) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return x;
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
        α = α_max;
        ++stepTooSmallCounter;
      } else {
        stepTooSmallCounter = 0;
      }

      // xₖ₊₁ = xₖ + αₖpₖˣ
      // sₖ₊₁ = xₖ + αₖpₖˢ
      // yₖ₊₁ = xₖ + αₖᶻpₖʸ
      // zₖ₊₁ = xₖ + αₖᶻpₖᶻ
      x += α * p_x;
      s += α * p_s;
      y += α_z * p_y;
      z += α_z * p_z;

      // A requirement for the convergence proof is that the "primal-dual
      // barrier term Hessian" Σₖ does not deviate arbitrarily much from the
      // "primal Hessian" μⱼSₖ⁻². We ensure this by resetting
      //
      //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
      //
      // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
      {
        // Barrier parameter scale factor for inequality constraint Lagrange
        // multiplier safeguard
        constexpr double κ_Σ = 1e10;

        for (int row = 0; row < z.rows(); ++row) {
          z(row) =
              std::max(std::min(z(row), κ_Σ * μ / s(row)), μ / (κ_Σ * s(row)));
        }
      }
    }

    // Update autodiff for Jacobians and Hessian
    SetAD(xAD, x);
    SetAD(sAD, s);
    SetAD(yAD, y);
    SetAD(zAD, z);

    A_e = jacobianCe.Calculate();
    A_i = jacobianCi.Calculate();
    g = gradientF.Calculate();
    H = hessianL.Calculate();

    // Update cₑ
    for (int row = 0; row < c_e.rows(); ++row) {
      c_eAD(row).Update();
    }
    c_e = GetAD(c_eAD);

    // Update cᵢ
    for (int row = 0; row < c_i.rows(); ++row) {
      c_iAD(row).Update();
    }
    c_i = GetAD(c_iAD);

    // Update the error estimate
    E_0 = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, 0.0);
    if (E_0 < acceptableTolerance) {
      ++acceptableIterCounter;
    } else {
      acceptableIterCounter = 0;
    }

    // Update the barrier parameter if necessary
    if (E_0 > config.tolerance) {
      // Barrier parameter scale factor for tolerance checks
      constexpr double κ_ε = 10.0;

      // While the error estimate is below the desired threshold for this
      // barrier parameter value, decrease the barrier parameter further
      double E_μ = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      while (μ > μ_min && E_μ <= κ_ε * μ) {
        UpdateBarrierParameterAndResetFilter();
        E_μ = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      }
    }

    auto innerIterEndTime = std::chrono::system_clock::now();

    if (config.diagnostics) {
      if (iterations % 20 == 0) {
        fmt::print("{:>4}   {:>10}  {:>10}   {:>16}  {:>19}\n", "iter",
                   "time (ms)", "error", "cost", "infeasibility");
        fmt::print("{:=^70}\n", "");
      }

      FilterEntry entry{f.value(), μ, s, c_e, c_i};
      fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations,
                 ToMilliseconds(innerIterEndTime - innerIterStartTime), E_0,
                 entry.cost, entry.constraintViolation);
    }

    ++iterations;
    if (iterations >= config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterationsExceeded;
      return x;
    }

    if (innerIterEndTime - solveStartTime > config.timeout) {
      status->exitCondition = SolverExitCondition::kMaxWallClockTimeExceeded;
      return x;
    }

    if (E_0 > config.tolerance &&
        acceptableIterCounter == maxAcceptableIterations) {
      status->exitCondition = SolverExitCondition::kSolvedToAcceptableTolerance;
      return x;
    }

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (stepTooSmallCounter == 2) {
      if (μ > μ_min) {
        UpdateBarrierParameterAndResetFilter();
        continue;
      } else {
        status->exitCondition =
            SolverExitCondition::kMaxSearchDirectionTooSmall;
        return x;
      }
    }
  }

  return x;
}

}  // namespace sleipnir
