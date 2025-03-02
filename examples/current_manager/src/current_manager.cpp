// Copyright (c) Sleipnir contributors

#include "current_manager.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

CurrentManager::CurrentManager(std::span<const double> current_tolerances,
                               double max_current)
    : m_desired_currents{static_cast<int>(current_tolerances.size()), 1},
      m_allocated_currents{
          m_problem.decision_variable(current_tolerances.size())} {
  // Ensure m_desired_currents contains initialized Variables
  for (int row = 0; row < m_desired_currents.rows(); ++row) {
    // Don't initialize to 0 or 1, because those will get folded by Sleipnir
    m_desired_currents[row] = std::numeric_limits<double>::infinity();
  }

  slp::Variable J = 0.0;
  slp::Variable current_sum = 0.0;
  for (size_t i = 0; i < current_tolerances.size(); ++i) {
    // The weight is 1/tolᵢ² where tolᵢ is the tolerance between the desired
    // and allocated current for subsystem i
    auto error = m_desired_currents[i] - m_allocated_currents[i];
    J += error * error / (current_tolerances[i] * current_tolerances[i]);

    current_sum += m_allocated_currents[i];

    // Currents must be nonnegative
    m_problem.subject_to(m_allocated_currents[i] >= 0.0);
  }
  m_problem.minimize(J);

  // Keep total current below maximum
  m_problem.subject_to(current_sum <= max_current);
}

std::vector<double> CurrentManager::calculate(
    std::span<const double> desired_currents) {
  if (m_desired_currents.rows() != static_cast<int>(desired_currents.size())) {
    throw std::runtime_error(
        "Number of desired currents must equal the number of tolerances "
        "passed in the constructor.");
  }

  for (size_t i = 0; i < desired_currents.size(); ++i) {
    m_desired_currents[i].set_value(desired_currents[i]);
  }

  m_problem.solve();

  std::vector<double> result;
  for (size_t i = 0; i < desired_currents.size(); ++i) {
    result.emplace_back(std::max(m_allocated_currents.value(i), 0.0));
  }

  return result;
}
