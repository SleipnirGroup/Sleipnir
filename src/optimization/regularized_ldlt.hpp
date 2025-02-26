// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "optimization/inertia.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Solves systems of linear equations using a regularized LDLT factorization.
 */
class RegularizedLDLT {
 public:
  /**
   * Constructs a RegularizedLDLT instance.
   */
  RegularizedLDLT() = default;

  /**
   * Reports whether previous computation was successful.
   *
   * @return Whether previous computation was successful.
   */
  Eigen::ComputationInfo info() const { return m_info; }

  /**
   * Computes the regularized LDLT factorization of a matrix.
   *
   * @param lhs Left-hand side of the system.
   * @param num_equality_constraints The number of equality constraints in the
   *   system.
   * @param μ The barrier parameter for the current interior-point iteration.
   * @return The factorization.
   */
  RegularizedLDLT& compute(const Eigen::SparseMatrix<double>& lhs,
                           size_t num_equality_constraints, double μ) {
    // The regularization procedure is based on algorithm B.1 of [1]
    m_num_decision_variables = lhs.rows() - num_equality_constraints;
    m_num_equality_constraints = num_equality_constraints;

    const Inertia ideal_inertia{m_num_decision_variables,
                                m_num_equality_constraints, 0};
    Inertia inertia;

    double δ = 0.0;
    double γ = 0.0;

    // Max density is 50% due to the caller only providing the lower triangle.
    // We consider less than 25% to be sparse.
    m_is_sparse = lhs.nonZeros() < 0.25 * lhs.size();

    if (m_is_sparse) {
      m_info = compute_sparse(lhs).info();
    } else {
      m_info = m_dense_solver.compute(lhs).info();
    }

    if (m_info == Eigen::Success) {
      if (m_is_sparse) {
        inertia = Inertia{m_sparse_solver};
      } else {
        inertia = Inertia{m_dense_solver};
      }

      // If the inertia is ideal, don't regularize the system
      if (inertia == ideal_inertia) {
        return *this;
      }
    }

    // If the decomposition succeeded and the inertia has some zero eigenvalues,
    // or the decomposition failed, regularize the equality constraints
    if ((m_info == Eigen::Success && inertia.zero > 0) ||
        m_info != Eigen::Success) {
      γ = 1e-8 * std::pow(μ, 0.25);
    }

    // Also regularize the Hessian. If the Hessian wasn't regularized in a
    // previous run of Compute(), start at a small value of δ. Otherwise,
    // attempt a δ half as big as the previous run so δ can trend downwards over
    // time.
    if (m_δ_old == 0.0) {
      δ = 1e-4;
    } else {
      δ = m_δ_old / 2.0;
    }

    while (true) {
      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + AᵢᵀΣAᵢ + δI   Aₑᵀ]
      //       [       Aₑ        −γI ]
      if (m_is_sparse) {
        m_info = compute_sparse(lhs + regularization(δ, γ)).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_sparse_solver};
        }
      } else {
        m_info = m_dense_solver.compute(lhs + regularization(δ, γ)).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_dense_solver};
        }
      }

      // If the inertia is ideal, store that value of δ and return.
      // Otherwise, increase δ by an order of magnitude and try again.
      if (m_info == Eigen::Success && inertia == ideal_inertia) {
        m_δ_old = δ;
        return *this;
      } else {
        δ *= 10.0;

        // If the Hessian perturbation is too high, report failure. This can be
        // caused by ill-conditioning.
        if (δ > 1e20) {
          m_info = Eigen::NumericalIssue;
          return *this;
        }
      }
    }
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& rhs) {
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs);
    }
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs.toDense());
    }
  }

  /**
   * Returns the Hessian regularization factor.
   *
   * @return Hessian regularization factor.
   */
  double hessian_regularization() const { return m_δ_old; }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  using DenseSolver = Eigen::LDLT<Eigen::MatrixXd>;

  SparseSolver m_sparse_solver;
  DenseSolver m_dense_solver;
  bool m_is_sparse = true;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  size_t m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  size_t m_num_equality_constraints = 0;

  /// The value of δ from the previous run of Compute().
  double m_δ_old = 0.0;

  // Number of non-zeros in LHS.
  int m_non_zeros = -1;

  /**
   * Computes factorization of a sparse matrix.
   *
   * @param lhs Matrix to factorize.
   * @return The factorization.
   */
  SparseSolver& compute_sparse(const Eigen::SparseMatrix<double>& lhs) {
    // Reanalize lhs's sparsity pattern if it changed
    int non_zeros = lhs.nonZeros();
    if (m_non_zeros != non_zeros) {
      m_sparse_solver.analyzePattern(lhs);
      m_non_zeros = non_zeros;
    }

    m_sparse_solver.factorize(lhs);

    return m_sparse_solver;
  }

  /**
   * Returns regularization matrix.
   *
   * @param δ The Hessian regularization factor.
   * @param γ The equality constraint Jacobian regularization factor.
   * @return Regularization matrix.
   */
  Eigen::SparseMatrix<double> regularization(double δ, double γ) {
    Eigen::VectorXd vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return Eigen::SparseMatrix<double>{vec.asDiagonal()};
  }
};

}  // namespace sleipnir
