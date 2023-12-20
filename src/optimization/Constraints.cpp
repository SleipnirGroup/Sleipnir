// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/Constraints.hpp"

#include <algorithm>

namespace sleipnir {

EqualityConstraints::operator bool() const {
  return std::all_of(
      constraints.begin(), constraints.end(),
      [](const auto& constraint) { return constraint.Value() == 0.0; });
}

InequalityConstraints::operator bool() const {
  return std::all_of(
      constraints.begin(), constraints.end(),
      [](const auto& constraint) { return constraint.Value() >= 0.0; });
}

}  // namespace sleipnir
