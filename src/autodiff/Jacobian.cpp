// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Jacobian.hpp"

using namespace sleipnir;

Jacobian::Jacobian(const VariableMatrix& variables,
                   const VariableMatrix& wrt) noexcept
    : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
  m_profiler.StartSetup();

  for (int row = 0; row < m_wrt.Rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  for (Variable variable : m_variables) {
    m_graphs.emplace_back(variable);
  }

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.Rows() * m_wrt.Rows() * 0.01);

  for (int row = 0; row < m_variables.Rows(); ++row) {
    if (m_variables(row).Type() == ExpressionType::kLinear) {
      // If the row is linear, compute its gradient once here and cache its
      // triplets. Constant rows are ignored because their gradients have no
      // nonzero triplets.
      m_graphs[row].ComputeAdjoints([&](int col, double adjoint) {
        m_cachedTriplets.emplace_back(row, col, adjoint);
      });
    } else if (m_variables(row).Type() > ExpressionType::kLinear) {
      // If the row is quadratic or nonlinear, add it to the list of nonlinear
      // rows to be recomputed in Calculate().
      m_nonlinearRows.emplace_back(row);
    }
  }

  for (int row = 0; row < m_wrt.Rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  if (m_nonlinearRows.empty()) {
    m_J.setFromTriplets(m_cachedTriplets.begin(), m_cachedTriplets.end());
  }

  m_profiler.StopSetup();
}

const Eigen::SparseMatrix<double>& Jacobian::Calculate() {
  if (m_nonlinearRows.empty()) {
    return m_J;
  }

  m_profiler.StartSolve();

  Update();

  // Copy the cached triplets so triplets added for the nonlinear rows are
  // thrown away at the end of the function
  auto triplets = m_cachedTriplets;

  // Compute each nonlinear row of the Jacobian
  for (int row : m_nonlinearRows) {
    m_graphs[row].ComputeAdjoints([&](int col, double adjoint) {
      triplets.emplace_back(row, col, adjoint);
    });
  }

  m_J.setFromTriplets(triplets.begin(), triplets.end());

  m_profiler.StopSolve();

  return m_J;
}

void Jacobian::Update() {
  for (auto& graph : m_graphs) {
    graph.Update();
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}
