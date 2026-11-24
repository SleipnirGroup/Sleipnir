// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Jacobian of a vector of expressions with respect
/// to a vector of variables.
///
/// The Jacobian is only recomputed if the expression is quadratic or higher
/// order, or if it's linear with coefficients that depend on parameters.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Jacobian {
 public:
  /// Constructs a Jacobian object.
  ///
  /// @param expression Expression of which to compute the Jacobian.
  /// @param wrt Variable with respect to which to compute the Jacobian.
  Jacobian(Real<Scalar> expression, Real<Scalar> wrt)
      : Jacobian{RealMatrix<Scalar>{std::move(expression)},
                 RealMatrix<Scalar>{std::move(wrt)}} {}

  /// Constructs a Jacobian object.
  ///
  /// @param expression Expression of which to compute the Jacobian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Jacobian.
  Jacobian(Real<Scalar> expression, SleipnirMatrixLike<Scalar> auto wrt)
      : Jacobian{RealMatrix<Scalar>{std::move(expression)}, std::move(wrt)} {}

  /// Constructs a Jacobian object.
  ///
  /// @param expressions Vector of expressions of which to compute the Jacobian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Jacobian.
  Jacobian(RealMatrix<Scalar> expressions, SleipnirMatrixLike<Scalar> auto wrt)
      : m_expressions{std::move(expressions)}, m_wrt{std::move(wrt)} {
    slp_assert(m_expressions.cols() == 1);
    slp_assert(m_wrt.cols() == 1);

    for (auto& expression : m_expressions) {
      m_top_lists.emplace_back(detail::topological_sort(expression.expr));
    }

    // Initialize column each expression's adjoint occupies in the Jacobian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt[col].expr->scratch = col;
    }

    // Make list of output rows as column-node pairs
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
      m_J.setFromTriplets(m_cached_triplets.begin(), m_cached_triplets.end());
    }
  }

  /// Returns the Jacobian as a RealMatrix.
  ///
  /// This is useful when constructing optimization problems with derivatives in
  /// them.
  ///
  /// @return The Jacobian as a RealMatrix.
  RealMatrix<Scalar> get() const {
    RealMatrix<Scalar> result{detail::empty, m_expressions.rows(),
                              m_wrt.rows()};

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

  /// Evaluates the Jacobian at wrt's value.
  ///
  /// @return The Jacobian at wrt's value.
  const Eigen::SparseMatrix<Scalar>& value() {
    if (m_nonlinear_rows.empty()) {
      return m_J;
    }

    for (auto& top_list : m_top_lists) {
      detail::update_values(top_list);
    }

    // Copy the cached triplets so triplets added for the nonlinear rows are
    // thrown away at the end of the function
    auto triplets = m_cached_triplets;

    // Compute each nonlinear row of the Jacobian
    for (int row : m_nonlinear_rows) {
      detail::append_triplets(m_top_lists[row], m_output_lists[row], triplets,
                              row);
    }

    m_J.setFromTriplets(triplets.begin(), triplets.end());

    return m_J;
  }

 private:
  RealMatrix<Scalar> m_expressions;
  RealMatrix<Scalar> m_wrt;

  /// List of topologically sorted graphs from parent to child, one for each row
  gch::small_vector<detail::ExpressionGraph<Scalar>> m_top_lists;

  /// List of output rows as column-node pairs
  gch::small_vector<
      gch::small_vector<std::pair<int, detail::Expression<Scalar>*>>>
      m_output_lists;

  Eigen::SparseMatrix<Scalar> m_J{m_expressions.rows(), m_wrt.rows()};

  /// Cached triplets for gradients of linear rows
  gch::small_vector<Eigen::Triplet<Scalar>> m_cached_triplets;

  /// List of row indices for nonlinear rows and linear rows with parameters
  /// whose gradients will be computed in value()
  gch::small_vector<int> m_nonlinear_rows;
};

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Jacobian<double>;

}  // namespace slp
