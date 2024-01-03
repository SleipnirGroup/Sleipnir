// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "optimization/solver/InteriorPoint.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/optimization/SolverConfig.hpp"
#include "sleipnir/optimization/SolverIterationInfo.hpp"
#include "sleipnir/optimization/SolverStatus.hpp"

namespace sleipnir {

/**
 * Finds the iterate that minimizes the constraint violation while not deviating
 * too far from the starting point. This is a fallback procedure when the normal
 * interior-point method fails to converge to a feasible point.
 *
 * @param[in] decisionVariables The list of decision variables.
 * @param[in] f The cost function.
 * @param[in] equalityConstraints The list of equality constraints.
 * @param[in] inequalityConstraints The list of inequality constraints.
 * @param[in] x The current iterate from the normal solve.
 * @param[in] s The current inequality constraint slack variables from the
 *   normal solve.
 * @param μ Barrier parameter.
 * @param[in] callback The user callback.
 * @param[in] config Configuration options for the solver.
 * @param[out] status The solver status.
 * @return The optimal state.
 */
inline Eigen::VectorXd FeasibilityRestoration(
    std::span<Variable> decisionVariables, Variable& f,
    std::span<Variable> equalityConstraints,
    std::span<Variable> inequalityConstraints, const Eigen::VectorXd& x,
    const Eigen::VectorXd& s, double μ,
    const std::function<bool(const SolverIterationInfo&)>& callback,
    const SolverConfig& config, SolverStatus* status) {
  // Feasibility restoration
  //
  //        min  Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - x_R)ᵀD_R(x - x_R)
  //         x
  //       pₑ,nₑ
  //       pᵢ,nᵢ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        cᵢ(x) - s - pᵢ + nᵢ = 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //        pᵢ ≥ 0
  //        nᵢ ≥ 0
  //
  // where ζ = √μ where μ is the barrier parameter, x_R is original iterate
  // before feasibility restoration, and D_R is a scaling matrix defined by
  //
  //   D_R = diag(min(1, 1/|x_R⁽¹⁾|), …, min(1, 1/|x_R|⁽ⁿ⁾)
  std::vector<Variable> fr_decisionVariables;
  fr_decisionVariables.reserve(decisionVariables.size() +
                               2 * equalityConstraints.size() +
                               2 * inequalityConstraints.size());

  // Assign x
  fr_decisionVariables.assign(decisionVariables.begin(),
                              decisionVariables.end());

  // Allocate pₑ, nₑ, pᵢ, and nᵢ
  for (size_t row = 0;
       row < 2 * equalityConstraints.size() + 2 * inequalityConstraints.size();
       ++row) {
    fr_decisionVariables.emplace_back();
  }

  VariableMatrix xAD{{&fr_decisionVariables[0], decisionVariables.size()}};

  VariableMatrix p_eq{{&fr_decisionVariables[decisionVariables.size()],
                       equalityConstraints.size()}};
  for (auto& p_i : p_eq) {
    p_i.SetValue(1.0);
  }
  VariableMatrix n_eq{{&fr_decisionVariables[decisionVariables.size() +
                                             equalityConstraints.size()],
                       equalityConstraints.size()}};
  for (auto& n_i : n_eq) {
    n_i.SetValue(1.0);
  }
  VariableMatrix p_ineq{{&fr_decisionVariables[decisionVariables.size() +
                                               2 * equalityConstraints.size()],
                         inequalityConstraints.size()}};
  for (auto& p_i : p_ineq) {
    p_i.SetValue(1.0);
  }
  VariableMatrix n_ineq{{&fr_decisionVariables[decisionVariables.size() +
                                               2 * equalityConstraints.size() +
                                               inequalityConstraints.size()],
                         inequalityConstraints.size()}};
  for (auto& n_i : n_ineq) {
    n_i.SetValue(1.0);
  }

  std::vector<Variable> fr_equalityConstraints;
  fr_equalityConstraints.assign(equalityConstraints.begin(),
                                equalityConstraints.end());
  for (size_t row = 0; row < fr_equalityConstraints.size(); ++row) {
    auto& constraint = fr_equalityConstraints[row];
    constraint = constraint - p_eq(row) + n_eq(row);
  }

  std::vector<Variable> fr_inequalityConstraints;
  fr_inequalityConstraints.assign(inequalityConstraints.begin(),
                                  inequalityConstraints.end());
  for (size_t row = 0; row < fr_inequalityConstraints.size(); ++row) {
    auto& constraint = fr_inequalityConstraints[row];
    constraint = constraint - s(row) - p_ineq(row) + n_ineq(row);
  }
  // Require p ≥ 0
  for (auto& p_i : p_eq) {
    fr_inequalityConstraints.emplace_back(p_i);
  }
  for (auto& p_i : p_ineq) {
    fr_inequalityConstraints.emplace_back(p_i);
  }
  // Require n ≥ 0
  for (auto& n_i : n_eq) {
    fr_inequalityConstraints.emplace_back(n_i);
  }
  for (auto& n_i : n_ineq) {
    fr_inequalityConstraints.emplace_back(n_i);
  }

  Variable J = 0.0;
  for (auto& p_i : p_eq) {
    J += p_i;
  }
  for (auto& p_i : p_ineq) {
    J += p_i;
  }
  for (auto& n_i : n_eq) {
    J += n_i;
  }
  for (auto& n_i : n_ineq) {
    J += n_i;
  }

  // D_R = diag(min(1, 1/|x_R⁽¹⁾|), …, min(1, 1/|x_R|⁽ⁿ⁾)
  Eigen::VectorXd D_R{x.rows()};
  for (int row = 0; row < D_R.rows(); ++row) {
    D_R(row) = std::min(1.0, 1.0 / std::abs(x(row)));
  }

  // ζ/2 (x - x_R)ᵀD_R(x - x_R)
  for (int row = 0; row < x.rows(); ++row) {
    J += std::sqrt(μ) / 2.0 * D_R(row) * sleipnir::pow(xAD(row) - x(row), 2);
  }

  return InteriorPoint(fr_decisionVariables, J, fr_equalityConstraints,
                       fr_inequalityConstraints, callback, config,
                       VariableMatrix{fr_decisionVariables}.Value(), true,
                       status)
      .segment(0, decisionVariables.size());
}

}  // namespace sleipnir
