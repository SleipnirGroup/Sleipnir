// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <span>
#include <utility>

#include <Eigen/Core>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>

#include "benchmark_util.hpp"
#include "sleipnir_autodiff_evaluator.hpp"

namespace {

/// Returns the extended Rosenbrock function.
///
/// @param x The input.
slp::Real<double> rosenbrock(const slp::RealMatrix<double>& x) {
  // Hessian sparsity pattern (x entry represents nonzero):
  //
  //        x₀    x₁    x₂    x₃  ⋯ xₙ₋₃  xₙ₋₂  xₙ₋₁
  //  x₀  [ x     x                                 ]
  //  x₁  [ x     x     x                           ]
  //  x₂  [       x     x     x                     ]
  //  ⋮   [                       ⋱                 ]
  // xₙ₋₂ [                          x     x     x  ]
  // xₙ₋₁ [                                x     x  ]
  slp::Real f = 0.0;
  for (int i = 0; i < x.rows() - 1; ++i) {
    slp::Real a = x[i + 1] - x[i] * x[i];
    slp::Real b = 1.0 - x[i];
    f += 100.0 * a * a + b * b;
  }
  return f;
}

}  // namespace

void register_sleipnir_benchmarks(std::span<const int> nums_variables) {
  register_autodiff_benchmarks("Sleipnir", nums_variables, [](int N) {
    slp::RealMatrix<double> x{N};
    slp::Hessian<double, Eigen::Lower> H{rosenbrock(x), x};
    return SleipnirAutodiffEvaluator{std::move(x), std::move(H)};
  });
}
