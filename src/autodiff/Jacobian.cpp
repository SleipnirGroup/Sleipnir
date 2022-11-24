// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Jacobian.hpp"

#include "sleipnir/autodiff/Gradient.hpp"

using namespace sleipnir::autodiff;

Jacobian::Jacobian(VectorXvar variables, VectorXvar wrt) noexcept
    : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
  m_profiler.StartSetup();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  for (Variable variable : m_variables) {
    m_graphs.emplace_back(variable);
  }

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  for (int row = 0; row < m_variables.rows(); ++row) {
    if (m_variables(row).expr->type == ExpressionType::kLinear) {
      // If the row is linear, compute its gradient once here and cache its
      // triplets. Constant rows are ignored because their gradients have no
      // nonzero triplets.
      ComputeRow(row, m_cachedTriplets);
    } else if (m_variables(row).expr->type > ExpressionType::kLinear) {
      // If the row is quadratic or nonlinear, add it to the list of nonlinear
      // rows to be recomputed in Calculate().
      m_nonlinearRows.emplace_back(row);
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
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

  for (int row : m_nonlinearRows) {
    ComputeRow(row, triplets);
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

void Jacobian::ComputeRow(int rowIndex,
                          std::vector<Eigen::Triplet<double>>& triplets) {
  auto& row = m_graphs[rowIndex];

  row.ComputeAdjoints([&](int row, double adjoint) {
    triplets.emplace_back(rowIndex, row, adjoint);
  });
}
