// Copyright (c) Sleipnir contributors

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/Variable.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindInequalityConstraints(nb::class_<InequalityConstraints>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<const std::vector<InequalityConstraints>&>(),
          "inequality_constraints"_a,
          DOC(sleipnir, InequalityConstraints, InequalityConstraints, 2));
  cls.def(
      "__bool__", [](InequalityConstraints& self) -> bool { return self; },
      nb::is_operator(), DOC(sleipnir, InequalityConstraints, operator, bool));
}

}  // namespace sleipnir
