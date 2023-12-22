// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Gradient.hpp"

using namespace sleipnir;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), VariableMatrix{wrt}} {}

Gradient::Gradient(Variable variable, const VariableMatrix& wrt) noexcept
    : m_jacobian{variable, wrt} {}

const Eigen::SparseVector<double>& Gradient::Calculate() {
  m_g = m_jacobian.Calculate();

  return m_g;
}

void Gradient::Update() {
  m_jacobian.Update();
}

Profiler& Gradient::GetProfiler() {
  return m_jacobian.GetProfiler();
}
