// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Gradient.hpp"

#include "sleipnir/IntrusiveSharedPtr.hpp"

using namespace sleipnir::autodiff;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variable{std::move(variable)}, m_wrt{wrt}, m_g{m_wrt.rows()} {
  m_profiler.StartSetup();
  if (m_variable.Type() == ExpressionType::kConstant) {
    // If the expression is constant, the gradient is zero.
    m_g.setZero();
  } else {
    m_graph = ExpressionGraph{*m_variable.expr};

    if (m_variable.expr->type == ExpressionType::kLinear) {
      // If the expression is linear, compute its gradient once here and cache
      // its value.
      Compute();
    }
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
  // Computes the gradient of the expression. Given the expression f and
  // variable x, the derivative df/dx is denoted the "adjoint" of x.
  Update();

  // Assigns leaf nodes their respective position in the gradient.
  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
  for (auto col : m_graph.GetList()) {
    col->adjoint = 0.0;
  }
  m_graph.GetList()[0]->adjoint = 1.0;

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
  // multiplied by dy/dx. If there are multiple "paths" from the root node to
  // variable; the variable's adjoint is the sum of each path's adjoint
  // contribution.
  for (auto col : m_graph.GetList()) {
    auto& lhs = col->args[0];
    auto& rhs = col->args[1];

    if (lhs != nullptr) {
      if (rhs != nullptr) {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, rhs->value, col->adjoint);
        rhs->adjoint +=
            col->gradientValueFuncs[1](lhs->value, rhs->value, col->adjoint);
      } else {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, 0.0, col->adjoint);
      }
    }

    // If variable is a leaf node, assign its adjoint to the gradient.
    if (col->row != -1) {
      m_g.coeffRef(col->row) = col->adjoint;
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }
}
