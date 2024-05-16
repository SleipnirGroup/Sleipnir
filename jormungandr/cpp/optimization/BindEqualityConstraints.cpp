// Copyright (c) Sleipnir contributors

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindEqualityConstraints(py::class_<EqualityConstraints>& cls) {
  cls.def(
      "__bool__", [](const EqualityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, EqualityConstraints, operator, bool));
}

}  // namespace sleipnir
