// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_dynamics_type(nb::enum_<DynamicsType>& e) {
  e.value("EXPLICIT_ODE", DynamicsType::EXPLICIT_ODE,
          DOC(slp, DynamicsType, EXPLICIT_ODE));
  e.value("DISCRETE", DynamicsType::DISCRETE, DOC(slp, DynamicsType, DISCRETE));
}

}  // namespace slp
