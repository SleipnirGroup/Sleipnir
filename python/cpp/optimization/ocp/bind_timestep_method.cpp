// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/ocp/timestep_method.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_timestep_method(nb::enum_<TimestepMethod>& e) {
  e.value("FIXED", TimestepMethod::FIXED, DOC(slp, TimestepMethod, FIXED));
  e.value("VARIABLE", TimestepMethod::VARIABLE,
          DOC(slp, TimestepMethod, VARIABLE));
  e.value("VARIABLE_SINGLE", TimestepMethod::VARIABLE_SINGLE,
          DOC(slp, TimestepMethod, VARIABLE_SINGLE));
}

}  // namespace slp
