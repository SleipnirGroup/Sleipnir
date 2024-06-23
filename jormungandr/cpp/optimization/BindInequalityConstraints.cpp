// Copyright (c) Sleipnir contributors

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindInequalityConstraints(py::class_<InequalityConstraints>& cls) {
  using namespace py::literals;

  cls.def(py::init<const std::vector<InequalityConstraints>&>(),
          "inequality_constraints"_a,
          DOC(sleipnir, InequalityConstraints, InequalityConstraints, 2));
  cls.def(
      "__bool__", [](InequalityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, InequalityConstraints, operator, bool));
}

}  // namespace sleipnir
