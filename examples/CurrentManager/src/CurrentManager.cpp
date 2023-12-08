// Copyright (c) Sleipnir contributors

#include "CurrentManager.hpp"

#include <algorithm>
#include <stdexcept>

CurrentManager::CurrentManager(
    std::span<const units::ampere_t> currentTolerances,
    units::ampere_t maxCurrent)
    : m_desiredCurrents{static_cast<int>(currentTolerances.size()), 1},
      m_allocatedCurrents{
          m_problem.DecisionVariable(currentTolerances.size())} {
  // Ensure m_desiredCurrents contains initialized Variables
  for (int row = 0; row < m_desiredCurrents.Rows(); ++row) {
    m_desiredCurrents(row) = 0.0;
  }

  sleipnir::Variable J = 0.0;
  sleipnir::Variable currentSum = 0.0;
  for (size_t i = 0; i < currentTolerances.size(); ++i) {
    // The weight is 1/tolᵢ² where tolᵢ is the tolerance between the desired
    // and allocated current for subsystem i
    auto error = m_desiredCurrents(i) - m_allocatedCurrents(i);
    J += error * error /
         (currentTolerances[i].value() * currentTolerances[i].value());

    currentSum += m_allocatedCurrents(i);

    // Currents must be nonnegative
    m_problem.SubjectTo(m_allocatedCurrents(i) >= 0);
  }
  m_problem.Minimize(J);

  // Keep total current below maximum
  m_problem.SubjectTo(currentSum <= maxCurrent.value());
}

std::vector<units::ampere_t> CurrentManager::Calculate(
    std::span<const units::ampere_t> desiredCurrents) {
  if (m_desiredCurrents.Rows() != static_cast<int>(desiredCurrents.size())) {
    throw std::runtime_error(
        "Number of desired currents must equal the number of tolerances "
        "passed in the constructor.");
  }

  for (size_t i = 0; i < desiredCurrents.size(); ++i) {
    m_desiredCurrents(i).SetValue(desiredCurrents[i].value());
  }

  m_problem.Solve();

  std::vector<units::ampere_t> result;
  for (size_t i = 0; i < desiredCurrents.size(); ++i) {
    result.emplace_back(std::max(m_allocatedCurrents.Value(i), 0.0));
  }

  return result;
}
