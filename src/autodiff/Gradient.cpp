// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Gradient.hpp"

using namespace sleipnir;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), VariableMatrix{wrt}} {}

Gradient::Gradient(Variable variable, const VariableMatrix& wrt) noexcept
    : m_jacobian{variable, wrt} {}

VariableMatrix Gradient::Get() const {
  return m_jacobian.Get();
}

const Eigen::SparseVector<double>& Gradient::Value() {
  m_g = m_jacobian.Value();

  return m_g;
}

void Gradient::Update() {
  m_jacobian.Update();
}

Profiler& Gradient::GetProfiler() {
  return m_jacobian.GetProfiler();
}
