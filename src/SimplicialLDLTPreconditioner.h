// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace sleipnir {

/**
 * A conjugate gradient preconditioner based on the simplicial LDLT
 * decomposition.
 */
template <typename _Scalar>
class SimplicialLDLTPreconditioner {
  using Scalar = _Scalar;
  using Matrix = Eigen::SparseMatrix<Scalar>;

 public:
  using StorageIndex = typename Matrix::StorageIndex;

  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic
  };

  constexpr SimplicialLDLTPreconditioner() = default;

  template <typename MatType>
  explicit SimplicialLDLTPreconditioner(const MatType& mat) : m_decomp{mat} {
    compute(mat);
  }

  constexpr Eigen::Index rows() const noexcept { return m_decomp.rows(); }
  constexpr Eigen::Index cols() const noexcept { return m_decomp.cols(); }

  template <typename MatType>
  SimplicialLDLTPreconditioner& analyzePattern(const MatType&) {
    return *this;
  }

  template <typename MatType>
  SimplicialLDLTPreconditioner& factorize(const MatType& mat) {
    m_decomp.compute(mat);

    m_isInitialized = true;

    return *this;
  }

  template <typename MatType>
  SimplicialLDLTPreconditioner& compute(const MatType& mat) {
    return factorize(mat);
  }

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_impl(const Rhs& b, Dest& x) const {
    x = m_decomp.solve(b);
  }

  template <typename Rhs>
  inline const Eigen::Solve<SimplicialLDLTPreconditioner, Rhs> solve(
      const Eigen::MatrixBase<Rhs>& b) const {
    eigen_assert(m_isInitialized &&
                 "SimplicialLDLTPreconditioner is not initialized.");
    eigen_assert(
        m_decomp.rows() == b.rows() &&
        "SimplicialLDLTPreconditioner::solve(): invalid number of rows of "
        "the right hand side matrix b");
    return Eigen::Solve<SimplicialLDLTPreconditioner, Rhs>(*this, b.derived());
  }

  Eigen::ComputationInfo info() { return Eigen::Success; }

 protected:
  Eigen::SimplicialLDLT<Matrix> m_decomp;
  bool m_isInitialized = false;
};

}  // namespace sleipnir
