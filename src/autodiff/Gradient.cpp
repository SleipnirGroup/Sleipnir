// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Gradient.hpp"

using namespace sleipnir;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_jacobian{MapVectorXvar(&variable, 1), std::move(wrt)} {}

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
