// Copyright (c) Joshua Nichols and Tyler Veness

#include "Util.h"

#include <fmt/core.h>
#include <sleipnir/optimization/OptimizationProblem.h>

int RunBenchmarksAndLog(
    bool diagnostics, units::second_t T, int minPower, int maxPower,
    std::function<casadi::Opti(units::second_t, int)> casadiSetup,
    std::function<sleipnir::OptimizationProblem(units::second_t, int)>
        sleipnirSetup) {
  std::ofstream results{"scalability-results.csv"};
  if (!results.is_open()) {
    return 1;
  }

  results << "Samples,"
          << "CasADi setup time (ms),CasADi solve time (ms),"
          << "Sleipnir setup time (ms),Sleipnir solve time (ms)\n";
  std::flush(results);

  std::vector<int> Ns;
  for (int power = minPower; power < maxPower; ++power) {
    for (int N = std::pow(10, power); N < std::pow(10, power + 1);
         N += std::pow(10, power)) {
      Ns.emplace_back(N);
    }
  }
  Ns.emplace_back(std::pow(10, maxPower));

  fmt::print("Solving from N = {} to N = {}.\n", Ns.front(), Ns.back());
  for (int N : Ns) {
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
