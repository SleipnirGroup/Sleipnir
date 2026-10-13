// Copyright (c) Sleipnir contributors

#pragma once

#include <span>
#include <utility>

#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <sleipnir/autodiff/variable_matrix.hpp>

/// Evaluates a Sleipnir autodiff object.
///
/// This wrapper exists for API consistency instead of necessity.
template <typename Derivative>
class SleipnirAutodiffEvaluator {
 public:
  /// Constructs a SleipnirAutodiffEvaluator.
  ///
  /// @param x The autodiff object's input.
  /// @param derivative The autodiff object.
  SleipnirAutodiffEvaluator(slp::VariableMatrix<double> x,
                            Derivative derivative)
      : m_x{std::move(x)}, m_derivative{std::move(derivative)} {}

  /// Evaluates the autodiff object at the given input.
  ///
  /// @param x The input.
  void evaluate(std::span<const double> x) {
    m_x.set_value(Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()));
    const double* values = m_derivative.value().valuePtr();
    benchmark::DoNotOptimize(values);
    benchmark::ClobberMemory();
  }

 private:
  slp::VariableMatrix<double> m_x;
  Derivative m_derivative;
};
