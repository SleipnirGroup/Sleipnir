// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <span>

#include <casadi/casadi.hpp>

#include "benchmark_util.hpp"
#include "casadi_autodiff_evaluator.hpp"

namespace {

/// Returns the extended Rosenbrock function.
///
/// @param x The input.
casadi::SX rosenbrock(const casadi::SX& x) {
  casadi::SX f = 0.0;
  for (int i = 0; i < x.size1() - 1; ++i) {
    casadi::SX a = x(i + 1) - x(i) * x(i);
    casadi::SX b = 1.0 - x(i);
    f += 100.0 * a * a + b * b;
  }
  return f;
}

}  // namespace

void register_casadi_benchmarks(std::span<const int> nums_variables) {
  register_autodiff_benchmarks("CasADi", nums_variables, [](int N) {
    auto x = casadi::SX::sym("x", N);
    return CasadiAutodiffEvaluator{
        casadi::Function{"g", {x}, {casadi::SX::gradient(rosenbrock(x), x)}}};
  });
}
