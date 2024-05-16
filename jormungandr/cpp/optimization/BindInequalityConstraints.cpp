// Copyright (c) Sleipnir contributors

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindInequalityConstraints(py::class_<InequalityConstraints>& cls) {
  cls.def(
      "__bool__",
      [](const InequalityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, InequalityConstraints, operator, bool));
}

}  // namespace sleipnir
