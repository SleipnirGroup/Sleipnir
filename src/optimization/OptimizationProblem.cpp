// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/OptimizationProblem.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <fmt/core.h>

#include "RegularizedLDLT.hpp"
#include "ScopeExit.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Variable.hpp"

using namespace sleipnir;

namespace {
/**
 * Filter entry consisting of objective value and constraint value.
 */
struct FilterEntry {
  /// The objective function's value
  double objective = 0.0;

  /// The constraint violation
  double constraintViolation = 0.0;

  constexpr FilterEntry() = default;

  /**
   * Constructs a FilterEntry.
   *
   * @param f The objective function.
   * @param mu The barrier parameter.
   * @param s The inequality constraint slack variables.
   * @param c_e The equality constraint values (nonzero means violation).
   * @param c_i The inequality constraint values (negative means violation).
   */
  FilterEntry(const Variable& f, double mu, Eigen::VectorXd& s,
              const Eigen::VectorXd& c_e, const Eigen::VectorXd& c_i)
      : objective{f.Value() - mu * s.array().log().sum()},
        constraintViolation{c_e.lpNorm<1>() + (c_i - s).lpNorm<1>()} {}
};

struct Filter {
  std::vector<FilterEntry> filter;

  double maxConstraintViolation;
  double minConstraintViolation;

  // double gamma_constraint = 1e-5;
  // double gamma_objective = 1e-5;
  double gamma_constraint = 0;
  double gamma_objective = 0;

  explicit Filter(FilterEntry pair) {
    filter.push_back(pair);
    minConstraintViolation = 1e-4 * std::max(1.0, pair.constraintViolation);
    maxConstraintViolation = 1e4 * std::max(1.0, pair.constraintViolation);
  }

  void PushBack(FilterEntry pair) { filter.push_back(pair); }

  void ResetFilter(FilterEntry pair) {
    filter.clear();
    filter.push_back(pair);
  }

  bool IsStepAcceptable(Eigen::VectorXd x, Eigen::VectorXd s,
                        Eigen::VectorXd p_x, Eigen::VectorXd p_s,
                        FilterEntry pair) {
    if (std::all_of(
            filter.begin(), filter.end(),
            [&](const auto& entry) {
              return pair.objective <=
                         entry.objective -
                             gamma_objective * entry.constraintViolation ||
                     pair.constraintViolation <=
                         (1 - gamma_constraint) * entry.constraintViolation;
            }) &&
        pair.constraintViolation < maxConstraintViolation) {
      return true;
    }
    return false;
  }
};

/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(std::vector<Variable>& dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.size() == static_cast<size_t>(src.rows()));

  for (size_t row = 0; row < dest.size(); ++row) {
    dest[row] = src(row);
  }
}

/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(Eigen::Ref<VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.rows() == src.rows());

  for (int row = 0; row < dest.rows(); ++row) {
    dest(row) = src(row);
  }
}

/**
 * Gets the contents of a autodiff vector as a double vector.
 *
 * @param src The autodiff vector.
 */
Eigen::VectorXd GetAD(std::vector<Variable> src) {
  Eigen::VectorXd dest{src.size()};
  for (int row = 0; row < dest.size(); ++row) {
    dest(row) = src[row].Value();
  }
  return dest;
}

Eigen::SparseMatrix<double> SparseDiagonal(const Eigen::VectorXd& src) {
  std::vector<Eigen::Triplet<double>> triplets;
  for (int row = 0; row < src.rows(); ++row) {
    triplets.emplace_back(row, row, src(row));
  }
  Eigen::SparseMatrix<double> dest{src.rows(), src.rows()};
  dest.setFromTriplets(triplets.begin(), triplets.end());
  return dest;
}

/**
 * Applies fraction-to-the-boundary rule to a variable and its iterate, then
 * returns a fraction of the iterate step size within (0, 1].
 *
 * @param x The variable.
 * @param p The iterate on the variable.
 * @param tau Fraction-to-the-boundary rule scaling factor.
 * @param max_alpha Maximum allowable step size.
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
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 * @param[in] transpose Whether to transpose mat.
 */
void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat,
                       bool transpose = false) {
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      if (transpose) {
        triplets.emplace_back(rowOffset + it.col(), colOffset + it.row(),
                              it.value());
      } else {
        triplets.emplace_back(rowOffset + it.row(), colOffset + it.col(),
                              it.value());
      }
    }
  }
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

void OptimizationProblem::Minimize(const Variable& cost) {
  m_f = cost;
}

void OptimizationProblem::Minimize(Variable&& cost) {
  m_f = std::move(cost);
}

void OptimizationProblem::Maximize(const Variable& objective) {
  // Maximizing an objective function is the same as minimizing its negative
  m_f = -objective;
}

void OptimizationProblem::Maximize(Variable&& objective) {
  // Maximizing an objective function is the same as minimizing its negative
  m_f = -std::move(objective);
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(std::move(storage[i]));
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

Eigen::VectorXd OptimizationProblem::InteriorPoint(
    const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
    SolverStatus* status) {
  // Let f(x)ₖ be the cost function, cₑ(x)ₖ be the equality constraints, and
  // cᵢ(x)ₖ be the inequality constraints. The Lagrangian of the optimization
  // problem is
  //
  //   L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  //
  // The Hessian of the Lagrangian is
  //
  //   H(x)ₖ = ∇²ₓₓL(x, s, y, z)ₖ
  //
  // The primal-dual barrier term Hessian Σ is defined as
  //
  //   Σ = S⁻¹Z
  //
  // where
  //
  //       [s₁ 0 ⋯ 0 ]
  //   S = [0  ⋱   ⋮ ]
  //       [⋮    ⋱ 0 ]
  //       [0  ⋯ 0 sₘ]
  //
  //       [z₁ 0 ⋯ 0 ]
  //   Z = [0  ⋱   ⋮ ]
  //       [⋮    ⋱ 0 ]
  //       [0  ⋯ 0 zₘ]
  //
  // and where m is the number of inequality constraints.
  //
  // Let f(x) = f(x)ₖ, H = H(x)ₖ, Aₑ = Aₑ(x)ₖ, and Aᵢ = Aᵢ(x)ₖ for clarity. We
  // want to solve the following Newton-KKT system shown in equation (19.12) of
  // [1].
  //
  //   [H    0  Aₑᵀ  Aᵢᵀ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  //   [0    Σ   0   −I ][ pₖˢ] = −[     z − μS⁻¹e     ]
  //   [Aₑ   0   0    0 ][−pₖʸ]    [        cₑ         ]
  //   [Aᵢ  −I   0    0 ][−pₖᶻ]    [      cᵢ − s       ]
  //
  // where e is a column vector of ones with a number of rows equal to the
  // number of inequality constraints.
  //
  // Solve the second row for pₖˢ.
  //
  //   Σpₖˢ + pₖᶻ = μS⁻¹e − z
  //   Σpₖˢ = μS⁻¹e − z − pₖᶻ
  //   pₖˢ = μΣ⁻¹S⁻¹e − Σ⁻¹z − Σ⁻¹pₖᶻ
  //
  // Substitute Σ = S⁻¹Z into the first two terms.
  //
  //   pₖˢ = μ(S⁻¹Z)⁻¹S⁻¹e − (S⁻¹Z)⁻¹z − Σ⁻¹pₖᶻ
  //   pₖˢ = μZ⁻¹SS⁻¹e − Z⁻¹Sz − Σ⁻¹pₖᶻ
  //   pₖˢ = μZ⁻¹e − s − Σ⁻¹pₖᶻ
  //
  // Substitute the explicit formula for pₖˢ into the fourth row and simplify.
  //
  //   Aᵢpₖˣ − pₖˢ = s − cᵢ
  //   Aᵢpₖˣ − (μZ⁻¹e − s − Σ⁻¹pₖᶻ) = s − cᵢ
  //   Aᵢpₖˣ − μZ⁻¹e + s + Σ⁻¹pₖᶻ = s − cᵢ
  //   Aᵢpₖˣ + Σ⁻¹pₖᶻ = −cᵢ + μZ⁻¹e
  //
  // Substitute the new second and fourth rows into the system.
  //
  //   [H   0  Aₑᵀ  Aᵢᵀ ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  //   [0   I   0    0  ][ pₖˢ] = −[−μZ⁻¹e + s + Σ⁻¹pₖᶻ]
  //   [Aₑ  0   0    0  ][−pₖʸ]    [        cₑ         ]
  //   [Aᵢ  0   0   −Σ⁻¹][−pₖᶻ]    [     cᵢ − μZ⁻¹e    ]
  //
  // Eliminate the second row and column.
  //
  //   [H   Aₑᵀ   Aᵢ ][ pₖˣ]    [∇f(x) − Aₑᵀy − Aᵢᵀz]
  //   [Aₑ   0    0  ][−pₖʸ] = −[        cₑ         ]
  //   [Aᵢ   0   −Σ⁻¹][−pₖᶻ]    [    cᵢ − μZ⁻¹e     ]
  //
  // Solve the third row for pₖᶻ.
  //
  //   Aₑpₖˣ + Σ⁻¹pₖᶻ = −cᵢ + μZ⁻¹e
  //   pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
  //
  // Substitute the explicit formula for pₖᶻ into the first row.
  //
  //   Hpₖˣ − Aₑᵀpₖʸ − Aᵢᵀpₖᶻ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  //   Hpₖˣ − Aₑᵀpₖʸ − Aᵢᵀ(−Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ) = −∇f(x) + Aₑᵀy + Aᵢᵀz
  //
  // Expand and simplify.
  //
  //   Hpₖˣ − Aₑᵀpₖʸ + AᵢᵀΣcᵢ − AᵢᵀμS⁻¹e + AᵢᵀΣAᵢpₖˣ = −∇f(x) + Aₑᵀy + Aᵢᵀz
  //   Hpₖˣ + AᵢᵀΣAᵢpₖˣ − Aₑᵀpₖʸ  = −∇f(x) + Aₑᵀy + AᵢᵀΣcᵢ + AᵢᵀμS⁻¹e + Aᵢᵀz
  //   (H + AᵢᵀΣAᵢ)pₖˣ − Aₑᵀpₖʸ = −∇f(x) + Aₑᵀy − Aᵢᵀ(Σcᵢ − μS⁻¹e − z)
  //   (H + AᵢᵀΣAᵢ)pₖˣ − Aₑᵀpₖʸ = −(∇f(x) − Aₑᵀy + Aᵢᵀ(Σcᵢ − μS⁻¹e − z))
  //
  // Substitute the new first and third rows into the system.
  //
  //   [H + AᵢᵀΣAᵢ   Aₑᵀ  0][ pₖˣ]    [∇f(x) − Aₑᵀy + Aᵢᵀ(Σcᵢ − μS⁻¹e − z)]
  //   [    Aₑ        0   0][−pₖʸ] = −[                cₑ                 ]
  //   [    0         0   I][−pₖᶻ]    [       −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ       ]
  //
  // Eliminate the third row and column.
  //
  //   [H + AᵢᵀΣAᵢ  Aₑᵀ][ pₖˣ] = −[∇f(x) − Aₑᵀy + Aᵢᵀ(Σcᵢ − μS⁻¹e − z)]
  //   [    Aₑ       0 ][−pₖʸ]    [                cₑ                 ]
  //
  // This reduced 2x2 block system gives the iterates pₖˣ and pₖʸ with the
  // iterates pₖᶻ and pₖˢ given by
  //
  //   pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
  //   pₖˢ = μZ⁻¹e − s − Σ⁻¹pₖᶻ
  //
  // The iterates are applied like so
  //
  //   xₖ₊₁ = xₖ + αₖᵐᵃˣpₖˣ
  //   sₖ₊₁ = xₖ + αₖᵐᵃˣpₖˢ
  //   yₖ₊₁ = xₖ + αₖᶻpₖʸ
  //   zₖ₊₁ = xₖ + αₖᶻpₖᶻ
  //
  // where αₖᵐᵃˣ and αₖᶻ are computed via the fraction-to-the-boundary rule
  // shown in equations (15a) and (15b) of [2].
  //
  //   αₖᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
  //         = max(α ∈ (0, 1] : αpₖˢ ≥ −τⱼsₖ)
  //   αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
  //       = max(α ∈ (0, 1] : αpₖᶻ ≥ −τⱼzₖ)
  //
  // [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
  //     Springer, 2006.
  // [2] Wächter, A. and Biegler, L. "On the implementation of an interior-point
  //     filter line-search algorithm for large-scale nonlinear programming",
  //     2005. http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf
  // [3] Byrd, R. and Nocedal J. and Waltz R. "KNITRO: An Integrated Package for
  //     Nonlinear Optimization", 2005.
  //     https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf

  auto solveStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    fmt::print("Number of equality constraints: {}\n",
               m_equalityConstraints.size());
    fmt::print("Number of inequality constraints: {}\n\n",
               m_inequalityConstraints.size());
  }

  // Barrier parameter scale factor κ_μ for tolerance checks
  constexpr double kappa_epsilon = 10.0;

  // Barrier parameter scale factor κ_Σ for inequality constraint Lagrange
  // multiplier safeguard
  constexpr double kappa_sigma = 1e10;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double tau_min = 0.99;

  // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
  constexpr double kappa_mu = 0.2;

  // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1, 2).
  constexpr double theta_mu = 1.5;

  // Barrier parameter μ
  double mu = 0.1;
  double old_mu = mu;

  // Fraction-to-the-boundary rule scale factor τ
  double tau = tau_min;

  std::vector<Eigen::Triplet<double>> triplets;

  Eigen::VectorXd x = initialGuess;
  MapVectorXvar xAD(m_decisionVariables.data(), m_decisionVariables.size());

  Eigen::VectorXd s = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  VectorXvar sAD = VectorXvar::Ones(m_inequalityConstraints.size());

  Eigen::VectorXd y = Eigen::VectorXd::Zero(m_equalityConstraints.size());
  VectorXvar yAD = VectorXvar::Zero(m_equalityConstraints.size());

  Eigen::VectorXd z = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  VectorXvar zAD = VectorXvar::Ones(m_inequalityConstraints.size());

  MapVectorXvar c_eAD(m_equalityConstraints.data(),
                      m_equalityConstraints.size());
  MapVectorXvar c_iAD(m_inequalityConstraints.data(),
                      m_inequalityConstraints.size());

  const Eigen::MatrixXd e = Eigen::VectorXd::Ones(s.rows());

  // L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  Variable L = m_f.value();
  if (m_equalityConstraints.size() > 0) {
    L -= yAD.transpose() * c_eAD;
  }
  if (m_inequalityConstraints.size() > 0) {
    L -= zAD.transpose() * (c_iAD - sAD);
  }
  ExpressionGraph graphL{L};

  Eigen::VectorXd step = Eigen::VectorXd::Zero(x.rows());

  SetAD(xAD, x);
  graphL.Update();

  Gradient gradientF{m_f.value(), xAD};
  Hessian hessianL{L, xAD};
  Jacobian jacobianCe{c_eAD, xAD};
  Jacobian jacobianCi{c_iAD, xAD};

  // Error estimate E_μ
  double E_mu = std::numeric_limits<double>::infinity();

  // Gradient of f ∇f
  Eigen::SparseVector<double> g = gradientF.Calculate();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
  Eigen::SparseMatrix<double> H = hessianL.Calculate();

  // Equality constraints cₑ
  Eigen::VectorXd c_e = GetAD(m_equalityConstraints);

  // Inequality constraints cᵢ
  Eigen::VectorXd c_i = GetAD(m_inequalityConstraints);

  Filter filter{FilterEntry(m_f.value(), mu, s, c_e, c_i)};

  // Equality constraint Jacobian Aₑ
  //
  //         [∇ᵀcₑ₁(x)ₖ]
  // Aₑ(x) = [∇ᵀcₑ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcₑₘ(x)ₖ]
  Eigen::SparseMatrix<double> A_e = jacobianCe.Calculate();

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(x)ₖ]
  // Aᵢ(x) = [∇ᵀcᵢ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(x)ₖ]
  Eigen::SparseMatrix<double> A_i = jacobianCi.Calculate();

  auto iterationsStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    // Print number of nonzeros in Lagrangian Hessian and constraint Jacobians
    std::string prints;

    if (status->costFunctionType <= ExpressionType::kQuadratic &&
        status->equalityConstraintType <= ExpressionType::kQuadratic &&
        status->inequalityConstraintType <= ExpressionType::kQuadratic) {
      prints += fmt::format("Number of nonzeros in Lagrangian Hessian: {}\n",
                            H.nonZeros());
    }
    if (status->equalityConstraintType <= ExpressionType::kLinear) {
      prints += fmt::format(
          "Number of nonzeros in equality constraint Jacobian: {}\n",
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

    fmt::print("Error tolerance: {}\n\n", m_config.tolerance);
  }

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

  RegularizedLDLT solver{theta_mu};

  while (E_mu > m_config.tolerance) {
    while (true) {
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

      // Check for problem local infeasibility. The problem is locally
      // infeasible if
      //
      //   Aₑᵀcₑ → 0
      //   Aᵢᵀcᵢ⁺ → 0
      //   ||(cₑ, cᵢ⁺)|| > ε
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
      if (m_inequalityConstraints.size() > 0) {
        Eigen::VectorXd c_i_plus = c_i.cwiseMin(0.0);
        if ((A_i.transpose() * c_i_plus).norm() < 1e-6 &&
            c_i_plus.norm() > 1e-6) {
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

          status->exitCondition = SolverExitCondition::kLocallyInfeasible;
          return x;
        }
      }

      // s_d = max(sₘₐₓ, (||y||₁ + ||z||₁) / (m + n)) / sₘₐₓ
      constexpr double s_max = 100.0;
      double s_d = std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) /
                                       (m_equalityConstraints.size() +
                                        m_inequalityConstraints.size())) /
                   s_max;

      // s_c = max(sₘₐₓ, ||z||₁ / n) / sₘₐₓ
      double s_c =
          std::max(s_max, z.lpNorm<1>() / m_inequalityConstraints.size()) /
          s_max;

      // Update the error estimate using the KKT conditions from equations
      // (19.5a) through (19.5d) in [1].
      //
      //   ∇f − Aₑᵀy − Aᵢᵀz = 0
      //   Sz − μe = 0
      //   cₑ = 0
      //   cᵢ − s = 0
      //
      // The error tolerance is the max of the following infinity norms scaled
      // by s_d and s_c (see equation (5) in [2]).
      //
      //   ||∇f − Aₑᵀy − Aᵢᵀz||_∞ / s_d
      //   ||Sz − μe||_∞ / s_c
      //   ||cₑ||_∞
      //   ||cᵢ − s||_∞
      Eigen::VectorXd eq1 = g;
      if (m_equalityConstraints.size() > 0) {
        eq1 -= A_e.transpose() * y;
      }
      if (m_inequalityConstraints.size() > 0) {
        eq1 -= A_i.transpose() * z;
      }
      E_mu = std::max(eq1.lpNorm<Eigen::Infinity>() / s_d,
                      (S * z - old_mu * e).lpNorm<Eigen::Infinity>() / s_c);
      if (m_equalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, c_e.lpNorm<Eigen::Infinity>());
      }
      if (m_inequalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, (c_i - s).lpNorm<Eigen::Infinity>());
      }

      if (E_mu <= kappa_epsilon * old_mu) {
        break;
      }

      //     [z₁ 0 ⋯ 0 ]
      // Z = [0  ⋱   ⋮ ]
      //     [⋮    ⋱ 0 ]
      //     [0  ⋯ 0 zₘ]
      Eigen::SparseMatrix<double> Z = SparseDiagonal(z);

      // Σ = S⁻¹Z
      Eigen::SparseMatrix<double> sigma = S.cwiseInverse() * Z;

      // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
      H = hessianL.Calculate();

      // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
      //       [    Aₑ       0 ]
      triplets.clear();
      Eigen::SparseMatrix<double> tmp = H;
      if (m_inequalityConstraints.size() > 0) {
        tmp += A_i.transpose() * sigma * A_i;
      }
      // Assign top-left quadrant
      AssignSparseBlock(triplets, 0, 0, tmp);
      if (m_equalityConstraints.size() > 0) {
        // Assign bottom-left quadrant
        AssignSparseBlock(triplets, tmp.rows(), 0, A_e);

        // Assign top-right quadrant
        AssignSparseBlock(triplets, 0, tmp.rows(), A_e, true);
      }
      Eigen::SparseMatrix<double> lhs{H.rows() + A_e.rows(),
                                      H.cols() + A_e.rows()};
      lhs.setFromTriplets(triplets.begin(), triplets.end());

      g = gradientF.Calculate();

      // rhs = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
      //        [               cₑ               ]
      //
      // The outer negative sign is applied in the solve() call.
      Eigen::VectorXd rhs{x.rows() + y.rows()};
      rhs.topRows(x.rows()) = g;
      if (m_equalityConstraints.size() > 0) {
        rhs.topRows(x.rows()) -= A_e.transpose() * y;
      }
      if (m_inequalityConstraints.size() > 0) {
        rhs.topRows(x.rows()) +=
            A_i.transpose() * (S.cwiseInverse() * (Z * c_i - mu * e) - z);
      }
      rhs.bottomRows(y.rows()) = c_e;

      // Solve the Newton-KKT system
      solver.Compute(lhs, m_equalityConstraints.size(), mu);
      step = solver.Solve(-rhs);

      // step = [ pₖˣ]
      //        [−pₖʸ]
      Eigen::VectorXd p_x = step.segment(0, x.rows());
      Eigen::VectorXd p_y = -step.segment(x.rows(), y.rows());

      // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
      Eigen::VectorXd p_z =
          -sigma * c_i + mu * S.cwiseInverse() * e - sigma * A_i * p_x;

      // pₖˢ = μZ⁻¹e − s − Σ⁻¹pₖᶻ
      Eigen::VectorXd p_s =
          mu * Z.cwiseInverse() * e - s - S * Z.cwiseInverse() * p_z;

      FilterEntry currentFilterEntry;

      bool stepAcceptable = false;

      double alpha_max = FractionToTheBoundaryRule(s, p_s, tau);

      // Apply second order corrections.
      Eigen::VectorXd p_x_soc = p_x;
      Eigen::VectorXd p_s_soc = p_s;
      Eigen::VectorXd p_y_soc, p_z_soc;
      for (int soc_iteration = 0; soc_iteration < 5; ++soc_iteration) {
        double alpha_soc = FractionToTheBoundaryRule(s, p_s_soc, tau);
        Eigen::VectorXd x_soc = x + alpha_soc * p_x_soc;
        Eigen::VectorXd s_soc = s + alpha_soc * p_s_soc;
        SetAD(xAD, x_soc);
        SetAD(sAD, s_soc);
        graphL.Update();
        c_e = GetAD(m_equalityConstraints);
        c_i = GetAD(m_inequalityConstraints);

        currentFilterEntry = FilterEntry(m_f.value(), mu, s_soc, c_e, c_i);
        if (filter.IsStepAcceptable(x, s, p_x, p_s, currentFilterEntry)) {
          p_x = p_x_soc;
          p_s = p_s_soc;
          alpha_max = alpha_soc;
          stepAcceptable = true;
          break;
        }

        // Rebuild Newton-KKT rhs with updated constraint values.
        rhs.topRows(x.rows()) = g;
        if (m_equalityConstraints.size() > 0) {
          rhs.topRows(x.rows()) -= A_e.transpose() * y;
        }
        if (m_inequalityConstraints.size() > 0) {
          rhs.topRows(x.rows()) +=
              A_i.transpose() *
              (SparseDiagonal(s_soc).cwiseInverse() * (Z * c_i - mu * e) - z);
        }
        rhs.bottomRows(y.rows()) = c_e;

        // Solve the Newton-KKT system
        step = solver.Solve(-rhs);

        p_x_soc = step.segment(0, x.rows());
        p_y_soc = -step.segment(x.rows(), y.rows());
        p_z_soc =
            -sigma * c_i + mu * S.cwiseInverse() * e - sigma * A_i * p_x_soc;
        p_s_soc =
            mu * Z.cwiseInverse() * e - s - sigma.cwiseInverse() * p_z_soc;
      }

      while (!stepAcceptable) {
        Eigen::VectorXd trial_x = x + alpha_max * p_x;
        Eigen::VectorXd trial_s = s + alpha_max * p_s;
        SetAD(xAD, trial_x);
        SetAD(sAD, trial_s);
        graphL.Update();

        c_e = GetAD(m_equalityConstraints);
        c_i = GetAD(m_inequalityConstraints);

        // If current filter entry is better than all prior ones in some
        // respect, accept it.
        currentFilterEntry = FilterEntry{m_f.value(), mu, trial_s, c_e, c_i};
        if (filter.IsStepAcceptable(x, s, p_x, p_s, currentFilterEntry)) {
          break;
        }
        alpha_max *= 0.5;
      }
      filter.PushBack(currentFilterEntry);

      // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
      double alpha_z = FractionToTheBoundaryRule(z, p_z, tau);

      // xₖ₊₁ = xₖ + αₖpₖˣ
      // sₖ₊₁ = xₖ + αₖpₖˢ
      // yₖ₊₁ = xₖ + αₖᶻpₖʸ
      // zₖ₊₁ = xₖ + αₖᶻpₖᶻ
      x += alpha_max * p_x;
      s += alpha_max * p_s;
      y += alpha_z * p_y;
      z += alpha_z * p_z;

      // A requirement for the convergence proof is that the "primal-dual
      // barrier term Hessian" Σₖ does not deviate arbitrarily much from the
      // "primal Hessian" μⱼSₖ⁻². We ensure this by resetting
      //
      //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
      //
      // for some fixed κ_Σ ≥ 1 after each step. See equation (16) in [2].
      for (int row = 0; row < z.rows(); ++row) {
        z(row) = std::max(std::min(z(row), kappa_sigma * mu / s(row)),
                          mu / (kappa_sigma * s(row)));
      }

      SetAD(xAD, x);
      SetAD(sAD, s);
      SetAD(yAD, y);
      SetAD(zAD, z);
      graphL.Update();

      auto innerIterEndTime = std::chrono::system_clock::now();

      if (m_config.diagnostics) {
        if (iterations % 20 == 0) {
          fmt::print("{:>4}   {:>10}  {:>10}   {:>16}  {:>19}\n", "iter",
                     "time (ms)", "error", "objective", "infeasibility");
          fmt::print("{:=^70}\n", "");
        }
        fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations,
                   ToMilliseconds(innerIterEndTime - innerIterStartTime), E_mu,
                   currentFilterEntry.objective,
                   currentFilterEntry.constraintViolation);
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
    }

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) in [2].
    old_mu = mu;
    mu = std::max(m_config.tolerance / 10.0,
                  std::min(kappa_mu * mu, std::pow(mu, theta_mu)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) in [2].
    tau = std::max(tau_min, 1.0 - mu);

    // Reset the filter when the barrier parameter is updated.
    filter.ResetFilter(FilterEntry(m_f.value(), mu, s, c_e, c_i));
  }

  return x;
}
