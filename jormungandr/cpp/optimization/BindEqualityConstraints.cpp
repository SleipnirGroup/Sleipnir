// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindEqualityConstraints(nb::class_<EqualityConstraints>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<const std::vector<EqualityConstraints>&>(),
          "equality_constraints"_a,
          DOC(sleipnir, EqualityConstraints, EqualityConstraints, 2));
  cls.def(
      "__bool__", [](EqualityConstraints& self) -> bool { return self; },
      nb::is_operator(), DOC(sleipnir, EqualityConstraints, operator, bool));
}

}  // namespace sleipnir
