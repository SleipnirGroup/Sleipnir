// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <span>
#include <utility>

#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "benchmark_util.hpp"
#include "sleipnir_autodiff_evaluator.hpp"

namespace {

/// Returns the test constraint function.
///
/// @param x The input.
slp::VariableMatrix<double> constraints(const slp::VariableMatrix<double>& x) {
  // Jacobian sparsity pattern (x entry represents nonzero):
  //
  //        x₀    x₁    x₂    x₃  ⋯ xₙ₋₃  xₙ₋₂  xₙ₋₁
  //  c₀  [ x     x                                 ]
  //  c₁  [       x     x                           ]
  //  c₂  [             x     x                     ]
  //  ⋮   [                       ⋱                 ]
  // cₙ₋₃ [                          x     x        ]
  // cₙ₋₂ [                                x     x  ]
  slp::VariableMatrix<double> c{x.rows() - 1};
  for (int i = 0; i < x.rows() - 1; ++i) {
    c[i] = slp::sin(x[i]) * slp::exp(x[i + 1]) - x[i] * x[i + 1];
  }
  return c;
}

}  // namespace

void register_sleipnir_benchmarks(std::span<const int> nums_variables) {
  register_autodiff_benchmarks("Sleipnir", nums_variables, [](int N) {
    slp::VariableMatrix<double> x{N};
    slp::Jacobian J{constraints(x), x};
    return SleipnirAutodiffEvaluator{std::move(x), std::move(J)};
  });
}
