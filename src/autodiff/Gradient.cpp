// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Gradient.hpp"

#include "sleipnir/IntrusiveSharedPtr.hpp"

using namespace sleipnir::autodiff;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variable{std::move(variable)},
      m_wrt{wrt},
      m_graph{m_variable},
      m_g{m_wrt.rows()} {
  m_profiler.StartSetup();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  m_graph = ExpressionGraph(m_variable);

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  if (m_variable.Type() == ExpressionType::kConstant) {
    // If the expression is constant, the gradient is zero.
    m_g.setZero();
  } else if (m_variable.expr->type == ExpressionType::kLinear) {
    // If the expression is linear, compute its gradient once here and cache its
    // value.
    Compute();
  }
  m_profiler.StopSetup();
}

const Eigen::SparseVector<double>& Gradient::Calculate() {
  if (m_variable.Type() > ExpressionType::kLinear) {
    m_profiler.StartSolve();
    Compute();
    m_profiler.StopSolve();
  }

  return m_g;
}

void Gradient::Update() {
  m_graph.Update();
}

Profiler& Gradient::GetProfiler() {
  return m_profiler;
}

void Gradient::Compute() {
  Update();
  m_graph.ComputeAdjoints(
      [&](int row, double adjoint) { m_g.coeffRef(row) = adjoint; });
}
