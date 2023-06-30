// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "Inertia.hpp"
#include "sleipnir/util/SparseMatrixBuilder.hpp"
#include "sleipnir/util/SparseUtil.hpp"

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
   */
  Eigen::ComputationInfo Info() { return m_info; }

  /**
   * Computes the regularized LDLT factorization of a matrix.
   *
   * @param lhs Left-hand side of the system.
   * @param numEqualityConstraints The number of equality constraints in the
   *   system.
   * @param mu The barrier parameter for the current interior-point iteration.
   */
  void Compute(const Eigen::SparseMatrix<double>& lhs,
               size_t numEqualityConstraints, double mu) {
    // The regularization procedure is based on algorithm B.1 of [1].
    //
    // [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed.,
    //     App. B. Springer, 2006.
    m_numDecisionVariables = lhs.rows() - numEqualityConstraints;
    m_numEqualityConstraints = numEqualityConstraints;

    const Inertia idealInertia{m_numDecisionVariables, m_numEqualityConstraints,
                               0};

    double delta = 0.0;
    double gamma = 0.0;

    m_solver.compute(lhs);
    Inertia inertia{m_solver};

    // If the decomposition succeeded and the inertia is ideal, don't regularize
    // the system
    if (m_solver.info() == Eigen::Success && inertia == idealInertia) {
      m_info = Eigen::Success;
      return;
    }

    // If the decomposition succeeded and the inertia has some zero eigenvalues,
    // or the decomposition failed, regularize the equality constraints and try
    // again
    if ((m_solver.info() == Eigen::Success && inertia.zero > 0) ||
        m_solver.info() != Eigen::Success) {
      gamma = 1e-8 * std::pow(mu, 0.25);

      m_solver.compute(lhs + Regularization(delta, gamma));
      inertia = Inertia{m_solver};

      if (m_solver.info() == Eigen::Success && inertia == idealInertia) {
        m_info = Eigen::Success;
        return;
      }
    }

    // Since adding gamma didn't fix the inertia, the Hessian needs to be
    // regularized. If the Hessian wasn't regularized in a previous run of
    // Compute(), start at a small value of delta. Otherwise, attempt a delta
    // half as big as the previous run so delta can trend downwards over time.
    if (m_deltaOld == 0.0) {
      delta = 1e-4;
    } else {
      delta = m_deltaOld / 2.0;
    }

    while (true) {
      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + AᵢᵀΣAᵢ + δI   Aₑᵀ]
      //       [       Aₑ        −γI ]
      m_solver.compute(lhs + Regularization(delta, gamma));
      Inertia inertia{m_solver};

      // If the inertia is ideal, store that value of delta and return.
      // Otherwise, increase delta by an order of magnitude and try again.
      if (inertia == idealInertia) {
        m_deltaOld = delta;
        m_info = Eigen::Success;
        return;
      } else {
        delta *= 10.0;

        // If the Hessian perturbation is too high, report failure. This can
        // happen due to a rank-deficient equality constraint Jacobian with
        // linearly dependent constraints.
        if (delta > 1e20) {
          m_info = Eigen::NumericalIssue;
          return;
        }
      }
    }
  }

  /**
   * Solve the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   */
  template <typename Rhs>
  auto Solve(const Eigen::MatrixBase<Rhs>& rhs) {
    return m_solver.solve(rhs);
  }

 private:
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> m_solver;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  size_t m_numDecisionVariables = 0;

  /// The number of equality constraints in the system.
  size_t m_numEqualityConstraints = 0;

  /// The value of delta from the previous run of Compute().
  double m_deltaOld = 0.0;

  /**
   * Returns regularization matrix.
   *
   * @param delta The Hessian regularization factor.
   * @param gamma The equality constraint Jacobian regularization factor.
   */
  Eigen::SparseMatrix<double> Regularization(double delta, double gamma) {
    int rows = m_numDecisionVariables + m_numEqualityConstraints;

    SparseMatrixBuilder<double> reg{rows, rows};
    reg.Block(0, 0, m_numDecisionVariables, m_numDecisionVariables) =
        delta * SparseIdentity(m_numDecisionVariables, m_numDecisionVariables);
    reg.Block(m_numDecisionVariables, m_numDecisionVariables,
              m_numEqualityConstraints, m_numEqualityConstraints) =
        -gamma *
        SparseIdentity(m_numEqualityConstraints, m_numEqualityConstraints);
    return reg.Build();
  }
};

}  // namespace sleipnir
