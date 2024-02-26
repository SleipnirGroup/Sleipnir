// Copyright (c) Sleipnir contributors

#pragma once

#include <span>
#include <vector>

#include <sleipnir/optimization/OptimizationProblem.hpp>

/**
 * This class computes the optimal current allocation for a list of subsystems
 * given a list of their desired currents and current tolerances that determine
 * which subsystem gets less current if the current budget is exceeded.
 * Subsystems with a smaller tolerance are given higher priority.
 */
class CurrentManager {
 public:
  /**
   * Constructs a CurrentManager.
   *
   * @param currentTolerances The relative current tolerance of each subsystem.
   * @param maxCurrent The current budget to allocate between subsystems.
   */
  CurrentManager(std::span<const double> currentTolerances, double maxCurrent);

  /**
   * Returns the optimal current allocation for a list of subsystems given a
   * list of their desired currents and current tolerances that determine which
   * subsystem gets less current if the current budget is exceeded. Subsystems
   * with a smaller tolerance are given higher priority.
   *
   * @param desiredCurrents The desired current for each subsystem.
   * @throws std::runtime_error if the number of desired currents doesn't equal
   *         the number of tolerances passed in the constructor.
   */
  std::vector<double> Calculate(std::span<const double> desiredCurrents);

 private:
  sleipnir::OptimizationProblem m_problem;
  sleipnir::VariableMatrix m_desiredCurrents;
  sleipnir::VariableMatrix m_allocatedCurrents;
};
