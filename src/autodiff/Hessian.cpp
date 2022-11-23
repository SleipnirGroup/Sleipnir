// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Hessian.hpp"

#include "sleipnir/autodiff/ExpressionGraph.hpp"

using namespace sleipnir::autodiff;

Hessian::Hessian(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_jacobian{ExpressionGraph{variable}.GenerateGradientTree(wrt), wrt} {}

const Eigen::SparseMatrix<double>& Hessian::Calculate() {
  return m_jacobian.Calculate();
}

void Hessian::Update() {
  m_jacobian.Update();
}

Profiler& Hessian::GetProfiler() {
  return m_jacobian.GetProfiler();
}
