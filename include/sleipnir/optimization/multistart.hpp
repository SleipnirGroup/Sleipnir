// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <future>
#include <span>

#include "sleipnir/optimization/solver_status.hpp"
#include "sleipnir/util/function_ref.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * The result of a multistart solve.
 *
 * @tparam DecisionVariables The type containing the decision variable initial
 *   guess.
 */
template <typename DecisionVariables>
struct MultistartResult {
  /// The solver status.
  SolverStatus status;
  /// The decision variables.
  DecisionVariables variables;
};

/**
 * Solves an optimization problem from different starting points in parallel,
 * then returns the solution with the lowest cost.
 *
 * Each solve is performed on a separate thread. Solutions from successful
 * solves are always preferred over solutions from unsuccessful solves, and cost
 * (lower is better) is the tiebreaker between successful solves.
 *
 * @tparam DecisionVariables The type containing the decision variable initial
 *   guess.
 * @param solve A user-provided function that takes a decision variable initial
 *   guess and returns a MultistartResult.
 * @param initial_guesses A list of decision variable initial guesses to try.
 */
template <typename DecisionVariables>
MultistartResult<DecisionVariables> Multistart(
    function_ref<MultistartResult<DecisionVariables>(
        const DecisionVariables& initial_guess)>
        solve,
    std::span<const DecisionVariables> initial_guesses) {
  small_vector<std::future<MultistartResult<DecisionVariables>>> futures;
  futures.reserve(initial_guesses.size());

  for (const auto& initial_guess : initial_guesses) {
    futures.emplace_back(std::async(std::launch::async, solve, initial_guess));
  }

  small_vector<MultistartResult<DecisionVariables>> results;
  results.reserve(futures.size());

  for (auto& future : futures) {
    results.emplace_back(future.get());
  }

  return *std::ranges::min_element(results, [](const auto& a, const auto& b) {
    // Prioritize successful solve
    if (a.status.exit_condition == SolverExitCondition::SUCCESS &&
        b.status.exit_condition != SolverExitCondition::SUCCESS) {
      return true;
    }

    // Otherwise prioritize solution with lower cost
    return a.status.cost < b.status.cost;
  });
}

}  // namespace sleipnir
