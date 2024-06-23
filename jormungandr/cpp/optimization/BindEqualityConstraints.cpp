// Copyright (c) Sleipnir contributors

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindEqualityConstraints(py::class_<EqualityConstraints>& cls) {
  using namespace py::literals;

  cls.def(py::init<const std::vector<EqualityConstraints>&>(),
          "equality_constraints"_a,
          DOC(sleipnir, EqualityConstraints, EqualityConstraints, 2));
  cls.def(
      "__bool__", [](EqualityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, EqualityConstraints, operator, bool));
}

}  // namespace sleipnir
