// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace sleipnir {

/**
 * Represents the inertia of a matrix (the number of positive, negative, and
 * zero eigenvalues).
 */
class Inertia {
 public:
  size_t positive = 0;
  size_t negative = 0;
  size_t zero = 0;

  constexpr Inertia() = default;

  /**
   * Constructs the Inertia type with the given number of positive, negative,
   * and zero eigenvalues.
   *
   * @param positive The number of positive eigenvalues.
   * @param negative The number of negative eigenvalues.
   * @param zero The number of zero eigenvalues.
   */
  constexpr Inertia(size_t positive, size_t negative, size_t zero)
      : positive{positive}, negative{negative}, zero{zero} {}

  friend bool operator==(const Inertia& lhs, const Inertia& rhs) {
    return lhs.positive == rhs.positive && lhs.negative == rhs.negative &&
           lhs.zero == rhs.zero;
  }
};

/**
 * Solves systems of linear equations using a regularized LDLT factorization.
 */
class RegularizedLDLT {
 public:
  /**
   * Constructs a RegularizedLDLT instance.
   *
   * @param theta_mu Barrier parameter superlinear decrease power (1, 2)
   */
  explicit RegularizedLDLT(double theta_mu) : m_theta_mu{theta_mu} {}

  /**
   * Solve the system of equations using a regularized LDLT factorization.
   *
   * @param lhs Left-hand side of the system.
   * @param rhs Right-hand side of the system.
   * @param numEqualityConstraints The number of equality constraints in the
   *   system.
   * @param mu The barrier parameter for the current interior-point iteration.
   */
  template <typename Rhs>
  auto Solve(const Eigen::SparseMatrix<double>& lhs,
             const Eigen::MatrixBase<Rhs>& rhs, size_t numEqualityConstraints,
             double mu) {
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
      Inertia inertia = ComputeInertia(m_solver);
      if (inertia == idealInertia) {
        return m_solver.solve(rhs);
      }

      if (inertia.zero > 0) {
        gamma = 1e-8 * std::pow(mu, m_theta_mu);
      }
    } else {
      gamma = 1e-8 * std::pow(mu, m_theta_mu);
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
      //       [       Aₑ        −δI ]
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

      Inertia inertia = ComputeInertia(m_solver);
      if (inertia == idealInertia) {
        m_deltaOld = delta;
        return m_solver.solve(rhs);
      } else {
        delta *= 10.0;
      }
    }

    return m_solver.solve(rhs);
  }

 private:
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> m_solver;

  double m_theta_mu;

  double m_deltaOld = 0.0;

  std::vector<Eigen::Triplet<double>> m_triplets;

  static Inertia ComputeInertia(
      const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver) {
    Inertia inertia;

    auto D = solver.vectorD();
    for (int row = 0; row < D.rows(); ++row) {
      if (D(row) > 0.0) {
        ++inertia.positive;
      } else if (D(row) < 0.0) {
        ++inertia.negative;
      } else {
        ++inertia.zero;
      }
    }

    return inertia;
  }
};

}  // namespace sleipnir
