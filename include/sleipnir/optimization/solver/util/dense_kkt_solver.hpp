// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/inertia.hpp"

namespace slp {

/// Solves a dense KKT system.
///
/// Applies regularization so the solution is a descent direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class DenseKKTSolver {
 public:
  /// Type alias for dense matrix.
  using DenseMatrix = Eigen::MatrixX<Scalar>;
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;

  /// Constructs a DenseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  DenseKKTSolver(int num_decision_variables, int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Constructs a DenseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  /// @param γ_min The minimum constraint regularization.
  DenseKKTSolver(int num_decision_variables, int num_equality_constraints,
                 Scalar γ_min)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints},
        m_γ_min{γ_min} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the factorization of the KKT matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  DenseKKTSolver& compute(const DenseMatrix& lhs) {
    // Bunch-Kaufman pivoting is backward stable for indefinite matrices, so the
    // unregularized system is accepted whenever its inertia is ideal
    m_lhs = lhs;
    m_info = m_solver.compute(m_lhs).info();
    if (m_info == Eigen::Success && inertia() == ideal_inertia) {
      m_prev_δ = Scalar(0);
      m_prev_γ = Scalar(0);
      return *this;
    }

    // We'll give lhs the correct inertia by adding [δI, 0; 0, −γI] where δ and
    // γ regularize the Hessian and equality constraint Jacobian respectively.

    // The minimum nonzero Hessian regularization
    constexpr Scalar δ_min(1e-10);

    // The maximum constraint regularization. Larger values distort the equality
    // constraint Jacobian more than they help a rank-deficient one.
    constexpr Scalar γ_max(1e-8);

    // The initial nonzero constraint regularization
    constexpr Scalar γ_initial(1e-10);

    // If the previous δ was zero, attempt a small value. Otherwise, attempt a
    // smaller value than the previous δ so δ trends downward. δ has a floor
    // because when regularization is needed, δ is the only curvature in some
    // directions, so the KKT matrix's smallest eigenvalues shrink with δ. Too
    // small a δ makes the matrix ill-conditioned, which amplifies the
    // factorization's rounding error in the solution, and eventually puts those
    // eigenvalues within ε of zero, where the inertia check counts them as
    // zero.
    Scalar δ = m_prev_δ == Scalar(0) ? Scalar(1e-4)
                                     : std::max(m_prev_δ / Scalar(2), δ_min);

    // Start γ at the minimum to minimize equality constraint Jacobian
    // distortion.
    Scalar γ = m_γ_min;

    while (true) {
      m_lhs = lhs + regularization(δ, γ);
      m_info = m_solver.compute(m_lhs).info();

      if (m_info == Eigen::Success) {
        Inertia inertia = this->inertia();

        if (inertia == ideal_inertia) {
          // If the inertia is ideal, report success
          m_prev_δ = δ;
          m_prev_γ = γ;
          return *this;
        } else if (inertia.zero > 0) {
          if (γ == Scalar(0)) {
            // If there's zero eigenvalues and γ = 0, increase γ to potentially
            // compensate for a rank-deficient equality constraint Jacobian
            γ = γ_initial;
          } else {
            // If there's zero eigenvalues and γ > 0, increase δ and γ to drive
            // all eigenvalues away from zero
            δ *= Scalar(10);
            γ = std::min(γ * Scalar(10), γ_max);
          }
        } else if (inertia.negative > ideal_inertia.negative) {
          // If there's too many negative eigenvalues, increase δ to add more
          // positive eigenvalues
          δ *= Scalar(10);
        } else if (inertia.positive > ideal_inertia.positive) {
          // The KKT matrix has at least as many negative eigenvalues as
          // equality constraints if γ > 0 (from the −γI block) or if the
          // equality constraint Jacobian has full row rank (as in feasibility
          // restoration, where γ = 0). Too many positive eigenvalues can then
          // only come from rounding error, so increase δ and γ to drive all
          // eigenvalues away from zero.
          //
          // Unlike LDLT, Bunch-Kaufman pivoting avoids small pivots, so
          // there's no need to find which block the error came from.
          δ *= Scalar(10);
          γ = std::min(γ * Scalar(10), γ_max);
        }
      } else {
        // If the decomposition failed, increase δ and γ to drive all
        // eigenvalues away from zero
        δ *= Scalar(10);
        γ = γ == Scalar(0) ? γ_initial : std::min(γ * Scalar(10), γ_max);
      }

      // If the lhs perturbation is too high, report failure. This can be caused
      // by ill-conditioning.
      if (δ > Scalar(1e20) || γ > Scalar(1e20)) {
        m_info = Eigen::NumericalIssue;
        m_prev_δ = δ;
        m_prev_γ = γ;
        return *this;
      }
    }
  }

  /// Solves the system of equations.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::MatrixBase<Rhs>& rhs) const {
    DenseVector x = m_solver.solve(rhs);

    // The factorization is backward stable, but the matrix's conditioning
    // amplifies its rounding error in the solution, so apply iterative
    // refinement with the same factorization
    constexpr int max_refinement_steps = 3;
    for (int i = 0; i < max_refinement_steps; ++i) {
      DenseVector r = rhs - m_lhs.template selfadjointView<Eigen::Lower>() * x;
      if (r.norm() <= Scalar(1e-12) * rhs.norm()) {
        break;
      }
      x += m_solver.solve(r);
    }

    return x;
  }

  /// Solves the system of equations.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::SparseMatrixBase<Rhs>& rhs) const {
    return solve(rhs.toDense());
  }

  /// Returns the Hessian regularization factor.
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const { return m_prev_δ; }

  /// Returns the constraint Jacobian regularization factor.
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const { return m_prev_γ; }

 private:
  using Solver = Eigen::BunchKaufman<DenseMatrix>;

  Solver m_solver;

  /// The most recently factorized KKT matrix's lower triangle.
  DenseMatrix m_lhs;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The minimum constraint regularization.
  Scalar m_γ_min{1e-10};

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The value of δ from the previous run of compute().
  Scalar m_prev_δ{0};

  /// The value of γ from the previous run of compute().
  Scalar m_prev_γ{0};

  /// Returns the inertia of the most recently factorized KKT matrix.
  ///
  /// D from the Bunch-Kaufman factorization has 1x1 and 2x2 diagonal blocks.
  /// By Sylvester's law of inertia, the KKT matrix's inertia is the inertia of
  /// D, which is the inertia of the blocks' eigenvalues. Eigenvalues within ε
  /// of zero are counted as zero.
  ///
  /// @return The inertia of the most recently factorized KKT matrix.
  Inertia inertia() const {
    using std::abs;
    using std::hypot;

    constexpr Scalar ε = std::numeric_limits<Scalar>::epsilon();

    const auto D = m_solver.vectorD();
    const auto& D_sub = m_solver.subDiagonal();

    Inertia inertia;
    auto count = [&](Scalar λ) {
      if (λ > ε) {
        ++inertia.positive;
      } else if (λ < -ε) {
        ++inertia.negative;
      } else {
        ++inertia.zero;
      }
    };

    for (int k = 0; k < D.size(); ++k) {
      if (k + 1 < D.size() && D_sub[k] != Scalar(0)) {
        // The 2x2 block [a  b; b  c] has eigenvalues (a + c)/2 ± r where
        // r = √(((a − c)/2)² + b²)
        const Scalar a = D[k];
        const Scalar b = D_sub[k];
        const Scalar c = D[k + 1];
        const Scalar mid = (a + c) / Scalar(2);
        const Scalar r = hypot((a - c) / Scalar(2), abs(b));
        count(mid + r);
        count(mid - r);
        ++k;
      } else {
        count(D[k]);
      }
    }

    return inertia;
  }

  /// Returns regularization matrix.
  ///
  ///   [δI    0]
  ///   [ 0  −γI]
  ///
  /// @param δ The Hessian regularization factor.
  /// @param γ The equality constraint Jacobian regularization factor.
  /// @return Regularization matrix.
  DenseMatrix regularization(Scalar δ, Scalar γ) const {
    DenseVector vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return vec.asDiagonal().toDenseMatrix();
  }
};

}  // namespace slp
