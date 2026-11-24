// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/matrix.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Hessian of an expression with respect to a vector
/// of variables.
///
/// The gradient tree is cached so subsequent Hessian calculations are faster,
/// and the Hessian is only recomputed if the expression is nonlinear or its
/// quadratic coefficients depend on parameters.
///
/// @tparam Scalar Scalar type.
/// @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
///     Default is Lower | Upper.
template <typename Scalar, int UpLo>
  requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
class Hessian {
 public:
  /// Constructs a Hessian object.
  ///
  /// @param expression Expression of which to compute the Hessian.
  /// @param wrt Variable with respect to which to compute the Hessian.
  Hessian(Real<Scalar> expression, Real<Scalar> wrt)
      : Hessian{std::move(expression), Matrix<Scalar>{std::move(wrt)}} {}

  /// Constructs a Hessian object.
  ///
  /// @param expression Expression of which to compute the Hessian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Hessian.
  Hessian(Real<Scalar> expression, SleipnirMatrixLike<Scalar> auto wrt)
      : m_expressions{detail::gradient_tree(
            detail::topological_sort(expression.expr), wrt)},
        m_wrt{std::move(wrt)} {
    slp_assert(m_wrt.cols() == 1);

    for (auto& expression : m_expressions) {
      m_top_lists.emplace_back(detail::topological_sort(expression.expr));
    }

    // Initialize column each expression's adjoint occupies in the Jacobian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt[col].expr->scratch = col;
    }

    // Make list of only nodes in output row, and their output columns
    for (auto& top_list : m_top_lists) {
      m_output_lists.emplace_back();
      for (const auto& node : top_list) {
        if (node->scratch != -1) {
          m_output_lists.back().emplace_back(node->scratch, node);
        }
      }
    }

    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->scratch = -1;
    }

    for (int row = 0; row < m_expressions.rows(); ++row) {
      if (m_expressions[row].expr == nullptr) {
        continue;
      }

      if (detail::has_constant_gradient(m_top_lists[row])) {
        // If the row's gradient is constant, compute it once here and cache
        // its triplets
        detail::append_triplets(m_top_lists[row], m_output_lists[row],
                                m_cached_triplets, row);
      } else {
        // Otherwise, add it to the list of rows to be recomputed in value()
        m_nonlinear_rows.emplace_back(row);
      }
    }

    if (m_nonlinear_rows.empty()) {
      m_H.setFromTriplets(m_cached_triplets.begin(), m_cached_triplets.end());
      if constexpr (UpLo == Eigen::Lower) {
        m_H = m_H.template triangularView<Eigen::Lower>();
      }
    }
  }

  /// Returns the Hessian as a Matrix.
  ///
  /// This is useful when constructing optimization problems with derivatives in
  /// them.
  ///
  /// @return The Hessian as a Matrix.
  Matrix<Scalar> get() const {
    Matrix<Scalar> result{detail::empty, m_expressions.rows(), m_wrt.rows()};

    for (int row = 0; row < m_expressions.rows(); ++row) {
      auto grad = detail::gradient_tree(m_top_lists[row], m_wrt);
      for (int col = 0; col < m_wrt.rows(); ++col) {
        if (grad[col].expr != nullptr) {
          result[row, col] = std::move(grad[col]);
        } else {
          result[row, col] = Real{Scalar(0)};
        }
      }
    }

    return result;
  }

  /// Evaluates the Hessian at wrt's value.
  ///
  /// @return The Hessian at wrt's value.
  const Eigen::SparseMatrix<Scalar>& value() {
    if (m_nonlinear_rows.empty()) {
      return m_H;
    }

    for (auto& top_list : m_top_lists) {
      detail::update_values(top_list);
    }

    // Copy the cached triplets so triplets added for the nonlinear rows are
    // thrown away at the end of the function
    auto triplets = m_cached_triplets;

    // Compute each nonlinear row of the Hessian
    for (int row : m_nonlinear_rows) {
      detail::append_triplets(m_top_lists[row], m_output_lists[row], triplets,
                              row);
    }

    m_H.setFromTriplets(triplets.begin(), triplets.end());
    if constexpr (UpLo == Eigen::Lower) {
      m_H = m_H.template triangularView<Eigen::Lower>();
    }

    return m_H;
  }

 private:
  Matrix<Scalar> m_expressions;
  Matrix<Scalar> m_wrt;

  /// List of topologically sorted graphs from parent to child, one for each row
  gch::small_vector<detail::ExpressionGraph<Scalar>> m_top_lists;

  /// List of output rows as column-node pairs
  gch::small_vector<
      gch::small_vector<std::pair<int, detail::Expression<Scalar>*>>>
      m_output_lists;

  Eigen::SparseMatrix<Scalar> m_H{m_expressions.rows(), m_wrt.rows()};

  // Cached triplets for gradients of linear rows
  gch::small_vector<Eigen::Triplet<Scalar>> m_cached_triplets;

  // List of row indices for nonlinear rows and linear rows with parameters
  // whose gradients will be computed in value()
  gch::small_vector<int> m_nonlinear_rows;
};

// @cond Suppress Doxygen
extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Hessian<double, Eigen::Lower | Eigen::Upper>;
// @endcond

}  // namespace slp
