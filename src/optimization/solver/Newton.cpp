// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/Newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>

#include <Eigen/SparseCholesky>

#include "optimization/RegularizedLDLT.hpp"
#include "optimization/solver/util/ErrorEstimate.hpp"
#include "optimization/solver/util/Filter.hpp"
#include "optimization/solver/util/KKTError.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/ScopedProfiler.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/ScopeExit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/Print.hpp"
#include "util/PrintDiagnostics.hpp"
#endif

// See docs/algorithms.md#Works_cited for citation definitions.

namespace sleipnir {

void Newton(
    std::span<Variable> decisionVariables, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status) {
  const auto solveStartTime = std::chrono::steady_clock::now();

  small_vector<SetupProfiler> setupProfilers;
  setupProfilers.emplace_back("setup").Start();

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix xAD{decisionVariables};
  xAD.SetValue(x);

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ L setup").Start();

  // Lagrangian L
  //
  // L(xₖ, yₖ) = f(xₖ)
  auto L = f;

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇f(x) setup").Start();

  // Gradient of f ∇f
  Gradient gradientF{f, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇f(x) init solve").Start();

  Eigen::SparseVector<double> g = gradientF.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇²ₓₓL setup").Start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ)
  Hessian hessianL{L, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇²ₓₓL init solve").Start();

  Eigen::SparseMatrix<double> H = hessianL.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ precondition checks").Start();

  // Check whether initial guess has finite f(xₖ)
  if (!std::isfinite(f.Value())) {
    status->exitCondition =
        SolverExitCondition::kNonfiniteInitialCostOrConstraints;
    return;
  }

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ spy setup").Start();

  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::unique_ptr<Spy> H_spy;
  if (config.spy) {
    H_spy = std::make_unique<Spy>("H.spy", "Hessian", "Decision variables",
                                  "Decision variables", H.rows(), H.cols());
  }

  setupProfilers.back().Stop();

  int iterations = 0;

  Filter filter{f};

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  int acceptableIterCounter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setupProfilers[0].Stop();

  small_vector<SolveProfiler> solveProfilers;
  solveProfilers.emplace_back("solve");
  solveProfilers.emplace_back("  ↳ feasibility check");
  solveProfilers.emplace_back("  ↳ spy writes");
  solveProfilers.emplace_back("  ↳ user callbacks");
  solveProfilers.emplace_back("  ↳ linear system solve");
  solveProfilers.emplace_back("  ↳ line search");
  solveProfilers.emplace_back("  ↳ next iter prep");

  auto& innerIterProf = solveProfilers[0];
  auto& feasibilityCheckProf = solveProfilers[1];
  auto& spyWritesProf = solveProfilers[2];
  auto& userCallbacksProf = solveProfilers[3];
  auto& linearSystemSolveProf = solveProfilers[4];
  auto& lineSearchProf = solveProfilers[5];
  auto& nextIterPrepProf = solveProfilers[6];

  // Prints final diagnostics when the solver exits
  scope_exit exit{[&] {
    status->cost = f.Value();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      PrintTotalTime(iterations, setupProfilers[0], solveProfilers[0]);

      PrintSetupDiagnostics(setupProfilers);

      solveProfilers.push_back(gradientF.GetSolveProfiler());
      solveProfilers.back().name = "  ↳ ∇f(x)";
      solveProfilers.push_back(hessianL.GetSolveProfiler());
      solveProfilers.back().name = "  ↳ ∇²ₓₓL";
      PrintSolveDiagnostics(solveProfilers);
    }
#endif
  }};

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  if (config.diagnostics) {
    sleipnir::println("Error tolerance: {}\n", config.tolerance);
  }
#endif

  while (E_0 > config.tolerance &&
         acceptableIterCounter < config.maxAcceptableIterations) {
    ScopedProfiler innerIterProfiler{innerIterProf};
    ScopedProfiler feasibilityCheckProfiler{feasibilityCheckProf};

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite()) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return;
    }

    feasibilityCheckProfiler.Stop();
    ScopedProfiler spyWritesProfiler{spyWritesProf};

    // Write out spy file contents if that's enabled
    if (config.spy) {
      H_spy->Add(H);
    }

    spyWritesProfiler.Stop();
    ScopedProfiler userCallbacksProfiler{userCallbacksProf};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, Eigen::VectorXd::Zero(0), g, H,
                    Eigen::SparseMatrix<double>{},
                    Eigen::SparseMatrix<double>{}})) {
        status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
        return;
      }
    }

    userCallbacksProfiler.Stop();
    ScopedProfiler linearSystemSolveProfiler{linearSystemSolveProf};

    // rhs = −[∇f]
    Eigen::VectorXd rhs = -g;

    // Solve the Newton-KKT system
    //
    // [H][ pₖˣ] = −[∇f]
    solver.Compute(H, 0, config.tolerance / 10.0);
    Eigen::VectorXd step = solver.Solve(rhs);

    linearSystemSolveProfiler.Stop();
    ScopedProfiler lineSearchProfiler{lineSearchProf};

    // step = [ pₖˣ]
    Eigen::VectorXd p_x = step.segment(0, x.rows());

    constexpr double α_max = 1.0;
    double α = α_max;

    // Loop until a step is accepted. If a step becomes acceptable, the loop
    // will exit early.
    while (1) {
      Eigen::VectorXd trial_x = x + α * p_x;

      xAD.SetValue(trial_x);

      // If f(xₖ + αpₖˣ) isn't finite, reduce step size immediately
      if (!std::isfinite(f.Value())) {
        // Reduce step size
        α *= α_red_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.MakeEntry();
      if (filter.TryAdd(entry, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_red_factor;

      // Safety factor for the minimal step size
      constexpr double α_min_frac = 0.05;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report infeasible.
      if (α < α_min_frac * Filter::γConstraint) {
        double currentKKTError = KKTError(g);

        Eigen::VectorXd trial_x = x + α_max * p_x;

        // Upate autodiff
        xAD.SetValue(trial_x);

        double nextKKTError = KKTError(gradientF.Value());

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (nextKKTError <= 0.999 * currentKKTError) {
          α = α_max;

          // Accept step
          break;
        }

        status->exitCondition = SolverExitCondition::kLocallyInfeasible;
        return;
      }
    }

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
    }

    lineSearchProfiler.Stop();

    // xₖ₊₁ = xₖ + αₖpₖˣ
    x += α * p_x;

    // Update autodiff for Hessian
    xAD.SetValue(x);
    g = gradientF.Value();
    H = hessianL.Value();

    ScopedProfiler nextIterPrepProfiler{nextIterPrepProf};

    // Update the error estimate
    E_0 = ErrorEstimate(g);
    if (E_0 < config.acceptableTolerance) {
      ++acceptableIterCounter;
    } else {
      acceptableIterCounter = 0;
    }

    nextIterPrepProfiler.Stop();
    innerIterProfiler.Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      PrintIterationDiagnostics(iterations, IterationMode::kNormal,
                                innerIterProfiler.CurrentDuration(), E_0,
                                f.Value(), 0.0, solver.HessianRegularization(),
                                α, 0.0);
    }
#endif

    ++iterations;

    // Check for max iterations
    if (iterations >= config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterationsExceeded;
      return;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solveStartTime > config.timeout) {
      status->exitCondition = SolverExitCondition::kTimeout;
      return;
    }

    // Check for solve to acceptable tolerance
    if (E_0 > config.tolerance &&
        acceptableIterCounter == config.maxAcceptableIterations) {
      status->exitCondition = SolverExitCondition::kSolvedToAcceptableTolerance;
      return;
    }
  }
}

}  // namespace sleipnir
