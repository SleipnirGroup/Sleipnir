// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

/// Scalar absolute tolerance matcher.
template <typename T>
struct ScalarWithinAbsMatcher : Catch::Matchers::MatcherGenericBase {
  ScalarWithinAbsMatcher(T target, T margin)
      : target{std::move(target)}, margin{std::move(margin)} {}

  bool match(const T& matchee) const {
    using std::abs;
    return abs(target - matchee) <= margin;
  }

  std::string describe() const override {
    return std::format("is within {} of {}", margin, target);
  }

 private:
  T target;
  T margin;
};

/// Creates a scalar absolute tolerance matcher.
template <typename T>
ScalarWithinAbsMatcher<T> WithinAbs(T target, T margin) {
  return {std::move(target), std::move(margin)};
}

/// Returns the relative error |a − b|/max(|a|, |b|) in units of machine
/// epsilon, or zero if a and b are equal.
template <typename T>
T relative_error_in_eps(const T& a, const T& b) {
  using std::abs;
  using std::max;

  if (a == b) {
    return T(0);
  }
  return abs(a - b) / max(abs(a), abs(b)) / std::numeric_limits<T>::epsilon();
}

/// Scalar relative tolerance matcher.
///
/// Matches if |target − matchee| ≤ epsilon · max(|target|, |matchee|), but for
/// any scalar type.
template <typename T>
struct ScalarWithinRelMatcher : Catch::Matchers::MatcherGenericBase {
  ScalarWithinRelMatcher(T target, T epsilon)
      : target{std::move(target)}, epsilon{std::move(epsilon)} {}

  bool match(const T& matchee) const {
    using std::isnan;

    error = relative_error_in_eps(target, matchee);
    return !isnan(error) &&
           error <= epsilon / std::numeric_limits<T>::epsilon();
  }

  std::string describe() const override {
    return std::format("is within {}ε relative of {} (error is {}ε)",
                       epsilon / std::numeric_limits<T>::epsilon(), target,
                       error);
  }

 private:
  T target;
  T epsilon;
  mutable T error{0};
};

/// Creates a scalar relative tolerance matcher.
template <typename T>
ScalarWithinRelMatcher<T> WithinRel(T target, T epsilon) {
  return {std::move(target), std::move(epsilon)};
}

/// Scalar equality matcher.
template <typename T>
struct ScalarEqualsMatcher : Catch::Matchers::MatcherGenericBase {
  explicit ScalarEqualsMatcher(T target) : target{std::move(target)} {}

  bool match(const T& matchee) const { return matchee == target; }

  std::string describe() const override { return std::format("== {}", target); }

 private:
  T target;
};

/// Creates a scalar equality matcher.
///
/// Rejects types convertible to std::string (e.g., const char*) so Catch2's
/// std::string equality matcher is chosen instead.
template <typename T>
  requires(!std::convertible_to<const T&, std::string>)
ScalarEqualsMatcher<T> Equals(T target) {
  return ScalarEqualsMatcher<T>{std::move(target)};
}

/// Any Eigen::Matrix (i.e., not an expression template).
template <typename T>
concept DenseMatrix = std::same_as<
    T, Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime,
                     T::ColsAtCompileTime, T::Options, T::MaxRowsAtCompileTime,
                     T::MaxColsAtCompileTime>>;

/// Any Eigen::SparseMatrix (i.e., not an expression template).
template <typename T>
concept SparseMatrix =
    std::same_as<T, Eigen::SparseMatrix<typename T::Scalar, T::Options,
                                        typename T::StorageIndex>>;

/// Matrix absolute tolerance matcher.
template <typename Matrix>
  requires DenseMatrix<Matrix> || SparseMatrix<Matrix>
struct MatrixWithinAbsMatcher : Catch::Matchers::MatcherGenericBase {
  using Scalar = typename Matrix::Scalar;

  MatrixWithinAbsMatcher(Matrix target, Scalar margin)
      : target{std::move(target)}, margin{margin} {}

  bool match(const Matrix& matchee) const {
    using std::abs;
    using std::isnan;

    if (target.rows() != matchee.rows() || target.cols() != matchee.cols()) {
      return false;
    }

    Matrix error = target - matchee;

    if constexpr (DenseMatrix<Matrix>) {
      for (int row = 0; row < error.rows(); ++row) {
        for (int col = 0; col < error.cols(); ++col) {
          if (isnan(error(row, col)) || abs(error(row, col)) > margin) {
            return false;
          }
        }
      }
    } else {
      for (int col = 0; col < error.outerSize(); ++col) {
        for (typename Matrix::InnerIterator it{error, col}; it; ++it) {
          if (isnan(it.value()) || abs(it.value()) > margin) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /// Prevents implicit sparse-to-dense conversion of matchee.
  template <typename Derived>
    requires DenseMatrix<Matrix>
  bool match(const Eigen::SparseMatrixBase<Derived>& matchee) const = delete;

  /// Prevents implicit dense-to-sparse conversion of matchee.
  template <typename Derived>
    requires SparseMatrix<Matrix>
  bool match(const Eigen::DenseBase<Derived>& matchee) const = delete;

  std::string describe() const override {
    return std::format("\nis within {} elementwise of\n{}", margin,
                       Catch::StringMaker<Matrix>::convert(target));
  }

 private:
  Matrix target;
  Scalar margin;
};

/// Creates a dense matrix absolute tolerance matcher.
///
/// Uses a dynamic-size target so matchees with mismatched sizes fail the match
/// instead of Eigen producing a resize assertion.
template <typename Derived>
MatrixWithinAbsMatcher<Eigen::MatrixX<typename Derived::Scalar>> WithinAbs(
    const Eigen::DenseBase<Derived>& target, typename Derived::Scalar margin) {
  return {target, margin};
}

/// Creates a sparse matrix absolute tolerance matcher.
template <typename Derived>
MatrixWithinAbsMatcher<Eigen::SparseMatrix<typename Derived::Scalar>> WithinAbs(
    const Eigen::SparseMatrixBase<Derived>& target,
    typename Derived::Scalar margin) {
  return {target, margin};
}

/// Dense matrix relative tolerance matcher.
///
/// Matches if each element is within the scalar relative tolerance of the
/// corresponding target element. Zero target elements must match exactly, so
/// use WithinAbs() for matrices with zeros that have rounding error.
template <typename Matrix>
  requires DenseMatrix<Matrix>
struct MatrixWithinRelMatcher : Catch::Matchers::MatcherGenericBase {
  using Scalar = typename Matrix::Scalar;

  MatrixWithinRelMatcher(Matrix target, Scalar epsilon)
      : target{std::move(target)}, epsilon{epsilon} {}

  bool match(const Matrix& matchee) const {
    using std::isnan;
    using std::max;

    if (target.rows() != matchee.rows() || target.cols() != matchee.cols()) {
      return false;
    }

    error = Scalar(0);
    for (int row = 0; row < target.rows(); ++row) {
      for (int col = 0; col < target.cols(); ++col) {
        Scalar elem_error =
            relative_error_in_eps(target(row, col), matchee(row, col));
        if (isnan(elem_error)) {
          error = elem_error;
          return false;
        }
        error = max(error, elem_error);
      }
    }

    return error <= epsilon / std::numeric_limits<Scalar>::epsilon();
  }

  /// Prevents implicit sparse-to-dense conversion of matchee.
  template <typename Derived>
  bool match(const Eigen::SparseMatrixBase<Derived>& matchee) const = delete;

  std::string describe() const override {
    return std::format(
        "\nis within {}ε relative elementwise of\n{}\n(max error is {}ε)",
        epsilon / std::numeric_limits<Scalar>::epsilon(),
        Catch::StringMaker<Matrix>::convert(target), error);
  }

 private:
  Matrix target;
  Scalar epsilon;
  mutable Scalar error{0};
};

/// Creates a dense matrix relative tolerance matcher.
///
/// Uses a dynamic-size target so matchees with mismatched sizes fail the match
/// instead of Eigen producing a resize assertion.
template <typename Derived>
MatrixWithinRelMatcher<Eigen::MatrixX<typename Derived::Scalar>> WithinRel(
    const Eigen::DenseBase<Derived>& target, typename Derived::Scalar epsilon) {
  return {target, epsilon};
}
