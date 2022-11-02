// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <chrono>
#include <fstream>
#include <functional>

#include <casadi/casadi.hpp>
#include <sleipnir/optimization/OptimizationProblem.h>
#include <units/time.h>

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1000.0;
}

/**
 * Runs the setup and solve for an optimization problem instance, records the
 * setup time and solve time for each, then writes them to results.csv.
 *
 * @tparam Problem The optimization problem's type (casadi::Opti or
 *   sleipnir::OptimizationProblem).
 * @param results The results.csv file to which to write the results.
 * @param setup A function that returns an optimization problem instance.
 * @param solve A function that takes an optimization problem instance and
 *   solves it.
 */
template <typename Problem>
void RunBenchmark(std::ofstream& results, std::function<Problem()> setup,
                  std::function<void(Problem&)> solve) {
  // Record setup time
  auto setupStartTime = std::chrono::system_clock::now();
  auto problem = setup();
  auto setupEndTime = std::chrono::system_clock::now();

  results << ToMilliseconds(setupEndTime - setupStartTime);
  std::flush(results);

  results << ",";
  std::flush(results);

  // Record solve time
  auto solveStartTime = std::chrono::system_clock::now();
  solve(problem);
  auto solveEndTime = std::chrono::system_clock::now();

  results << ToMilliseconds(solveEndTime - solveStartTime);
  std::flush(results);
}

/**
 * Runs scalability benchmarks for CasADi and Sleipnir versions of an
 * optimization problem, records the setup time and solve time for each, then
 * writes them to results.csv.
 *
 * The scale of the problem is iteratively increased by increasing the number of
 * timesteps within the time horizon.
 *
 * @param diagnostics Whether to enable diagnostic prints.
 * @param T The time horizon of the optimization problem.
 * @param minPower The minimum power of 10 for the number of samples in the
 *   problem.
 * @param maxPower The maximum power of 10 for the number of samples in the
 *   problem.
 * @param casadiSetup A function that takes a time horizon and number of samples
 *   and returns a CasADi optimization problem instance.
 * @param sleipnirSetup A function that takes a time horizon and number of
 *   samples and returns a Sleipnir optimization problem instance.
 */
int RunBenchmarksAndLog(
    bool diagnostics, units::second_t T, int minPower, int maxPower,
    std::function<casadi::Opti(units::second_t, int)> casadiSetup,
    std::function<sleipnir::OptimizationProblem(units::second_t, int)>
        sleipnirSetup);
