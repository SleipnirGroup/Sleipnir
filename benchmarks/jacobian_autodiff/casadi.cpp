// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <span>

#include <casadi/casadi.hpp>

#include "benchmark_util.hpp"
#include "casadi_autodiff_evaluator.hpp"

namespace {

/// Returns the test constraint function.
///
/// @param x The input.
casadi::SX constraints(const casadi::SX& x) {
  // Jacobian sparsity pattern (x entry represents nonzero):
  //
  //        x₀    x₁    x₂    x₃  ⋯ xₙ₋₃  xₙ₋₂  xₙ₋₁
  //  c₀  [ x     x                                 ]
  //  c₁  [       x     x                           ]
  //  c₂  [             x     x                     ]
  //  ⋮   [                       ⋱                 ]
  // cₙ₋₃ [                          x     x        ]
  // cₙ₋₂ [                                x     x  ]
  casadi::SX c(x.size1() - 1, 1);
  for (int i = 0; i < x.size1() - 1; ++i) {
    c(i) = casadi::SX::sin(x(i)) * casadi::SX::exp(x(i + 1)) - x(i) * x(i + 1);
  }
  return c;
}

}  // namespace

void register_casadi_benchmarks(std::span<const int> nums_variables) {
  register_autodiff_benchmarks("CasADi", nums_variables, [](int N) {
    auto x = casadi::SX::sym("x", N);
    return CasadiAutodiffEvaluator{
        casadi::Function{"J", {x}, {casadi::SX::jacobian(constraints(x), x)}}};
  });
}
