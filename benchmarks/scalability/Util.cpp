// Copyright (c) Joshua Nichols and Tyler Veness

#include "Util.hpp"

#include <fmt/core.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

int RunBenchmarksAndLog(
    std::string_view filename, bool diagnostics, units::second_t T,
    std::span<int> sampleSizesToTest,
    std::function<casadi::Opti(units::second_t, int)> casadiSetup,
    std::function<sleipnir::OptimizationProblem(units::second_t, int)>
        sleipnirSetup) {
  std::ofstream results{std::string{filename}};
  if (!results.is_open()) {
    return 1;
  }

  results << "Samples,"
          << "CasADi setup time (ms),CasADi solve time (ms),"
          << "Sleipnir setup time (ms),Sleipnir solve time (ms)\n";
  std::flush(results);

  for (int N : sampleSizesToTest) {
    results << N << ",";
    std::flush(results);

    units::second_t dt = T / N;

    fmt::print(stderr, "CasADi (N = {})...", N);
    RunBenchmark<casadi::Opti>(
        results, [=] { return casadiSetup(dt, N); },
        [=](casadi::Opti& opti) {
          if (diagnostics) {
            opti.solver("ipopt");
          } else {
            opti.solver("ipopt", {{"print_time", 0}},
                        {{"print_level", 0}, {"sb", "yes"}});
          }
          opti.solve();
        });
    fmt::print(stderr, " done.\n");

    results << ",";
    std::flush(results);

    fmt::print(stderr, "Sleipnir (N = {})...", N);
    RunBenchmark<sleipnir::OptimizationProblem>(
        results, [=] { return sleipnirSetup(dt, N); },
        [=](sleipnir::OptimizationProblem& problem) {
          sleipnir::SolverConfig config;
          config.diagnostics = diagnostics;
          problem.Solve(config);
        });
    fmt::print(stderr, " done.\n");

    results << "\n";
    std::flush(results);
  }

  return 0;
}
