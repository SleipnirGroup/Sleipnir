// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "Inertia.hpp"
#include "sleipnir/optimization/SparseUtil.hpp"

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
    double delta = 0.0;
    double gamma = 0.0;

    size_t numDecisionVariables = lhs.rows() - numEqualityConstraints;

    Inertia idealInertia{lhs.rows() - numEqualityConstraints,
                         numEqualityConstraints, 0};

    m_solver.compute(lhs);
    if (m_solver.info() == Eigen::Success) {
      Inertia inertia{m_solver};
      if (inertia == idealInertia) {
        m_info = Eigen::Success;
        return;
      }

      if (inertia.zero > 0) {
        gamma = 1e-8 * std::pow(mu, 0.25);
      }
    } else {
      gamma = 1e-8 * std::pow(mu, 0.25);
    }

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
      Eigen::SparseMatrix<double> regularization{lhs.rows(), lhs.cols()};
      m_triplets.clear();
      AssignSparseBlock(
          m_triplets, 0, 0,
          delta * SparseIdentity(numDecisionVariables, numDecisionVariables));
      AssignSparseBlock(m_triplets, numDecisionVariables, numDecisionVariables,
                        -gamma * SparseIdentity(numEqualityConstraints,
                                                numEqualityConstraints));
      regularization.setFromTriplets(m_triplets.begin(), m_triplets.end());
      m_solver.compute(lhs + regularization);

      Inertia inertia{m_solver};
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

  double m_deltaOld = 0.0;

  std::vector<Eigen::Triplet<double>> m_triplets;
};

}  // namespace sleipnir
