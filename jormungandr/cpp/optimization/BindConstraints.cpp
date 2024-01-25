// Copyright (c) Sleipnir contributors

#include "optimization/BindConstraints.hpp"

#include <pybind11/operators.h>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindConstraints(py::module_& optimization) {
  py::class_<EqualityConstraints> equalityConstraints{
      optimization, "EqualityConstraints", DOC(sleipnir, EqualityConstraints)};
  equalityConstraints.def(
      "__bool__", [](const EqualityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, EqualityConstraints, operator, bool));

  py::class_<InequalityConstraints> inequalityConstraints{
      optimization, "InequalityConstraints",
      DOC(sleipnir_InequalityConstraints)};
  inequalityConstraints.def(
      "__bool__",
      [](const InequalityConstraints& self) -> bool { return self; },
      py::is_operator(), DOC(sleipnir, InequalityConstraints, operator, bool));
}

}  // namespace sleipnir
