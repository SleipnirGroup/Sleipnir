// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "Inertia.hpp"

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

    Inertia idealInertia{lhs.rows() - numEqualityConstraints,
                         numEqualityConstraints, 0};

    m_solver.compute(lhs);
    if (m_solver.info() == Eigen::Success) {
      Inertia inertia{m_solver};
      if (inertia == idealInertia) {
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
      for (size_t row = 0; row < lhs.rows() - numEqualityConstraints; ++row) {
        m_triplets.emplace_back(row, row, delta);
      }
      for (int row = lhs.rows() - numEqualityConstraints; row < lhs.rows();
           ++row) {
        m_triplets.emplace_back(row, row, -gamma);
      }
      regularization.setFromTriplets(m_triplets.begin(), m_triplets.end());
      m_solver.compute(lhs + regularization);

      Inertia inertia{m_solver};
      if (inertia == idealInertia) {
        m_deltaOld = delta;
        return;
      } else {
        delta *= 10.0;
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

  double m_deltaOld = 0.0;

  std::vector<Eigen::Triplet<double>> m_triplets;
};

}  // namespace sleipnir
