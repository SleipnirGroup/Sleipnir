// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Hessian.hpp"

#include "sleipnir/autodiff/ExpressionGraph.hpp"

using namespace sleipnir;

Hessian::Hessian(Variable variable, const VariableMatrix& wrt) noexcept
    : m_jacobian{detail::ExpressionGraph{variable}.GenerateGradientTree(wrt),
                 wrt} {}

const Eigen::SparseMatrix<double>& Hessian::Calculate() {
  return m_jacobian.Calculate();
}

void Hessian::Update() {
  m_jacobian.Update();
}

Profiler& Hessian::GetProfiler() {
  return m_jacobian.GetProfiler();
}
