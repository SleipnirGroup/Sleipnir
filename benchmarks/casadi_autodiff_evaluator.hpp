// Copyright (c) Sleipnir contributors

#pragma once

#include <span>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <casadi/casadi.hpp>

/// Evaluates a CasADi function with one input and one output.
///
/// Work buffers are preallocated so evaluations don't allocate.
class CasadiAutodiffEvaluator {
 public:
  /// Constructs a CasadiAutodiffEvaluator.
  ///
  /// @param function The CasADi function.
  explicit CasadiAutodiffEvaluator(casadi::Function function)
      : m_function{std::move(function)},
        m_arg(m_function.sz_arg()),
        m_res(m_function.sz_res()),
        m_iw(m_function.sz_iw()),
        m_w(m_function.sz_w()),
        m_output(m_function.nnz_out(0)),
        m_mem{m_function.checkout()} {}

  CasadiAutodiffEvaluator(const CasadiAutodiffEvaluator&) = delete;
  CasadiAutodiffEvaluator& operator=(const CasadiAutodiffEvaluator&) = delete;

  ~CasadiAutodiffEvaluator() { m_function.release(m_mem); }

  /// Evaluates the function at the given input.
  ///
  /// @param x The input.
  void evaluate(std::span<const double> x) {
    m_arg[0] = x.data();
    m_res[0] = m_output.data();
    m_function(m_arg.data(), m_res.data(), m_iw.data(), m_w.data(),
               static_cast<int>(m_mem));
    double* values = m_output.data();
    benchmark::DoNotOptimize(values);
    benchmark::ClobberMemory();
  }

 private:
  casadi::Function m_function;
  std::vector<const double*> m_arg;
  std::vector<double*> m_res;
  std::vector<casadi_int> m_iw;
  std::vector<double> m_w;
  std::vector<double> m_output;
  casadi_int m_mem;
};
