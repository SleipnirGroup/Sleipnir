// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <span>
#include <utility>

#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "benchmark_util.hpp"
#include "sleipnir_autodiff_evaluator.hpp"

namespace {

/// Returns the extended Rosenbrock function.
///
/// @param x The input.
slp::Variable<double> rosenbrock(const slp::VariableMatrix<double>& x) {
  slp::Variable f = 0.0;
  for (int i = 0; i < x.rows() - 1; ++i) {
    slp::Variable a = x[i + 1] - x[i] * x[i];
    slp::Variable b = 1.0 - x[i];
    f += 100.0 * a * a + b * b;
  }
  return f;
}

}  // namespace

void register_sleipnir_benchmarks(std::span<const int> nums_variables) {
  register_autodiff_benchmarks("Sleipnir", nums_variables, [](int N) {
    slp::VariableMatrix<double> x{N};
    slp::Gradient g{rosenbrock(x), x};
    return SleipnirAutodiffEvaluator{std::move(x), std::move(g)};
  });
}
