// Copyright (c) Sleipnir contributors

#include "autodiff/BindConstraints.hpp"

#include <pybind11/operators.h>
#include <sleipnir/optimization/Constraints.hpp>

namespace py = pybind11;

namespace sleipnir {

void BindConstraints(py::module_& optimization) {
  py::class_<EqualityConstraints> equalityConstraints{optimization,
                                                      "EqualityConstraints"};
  equalityConstraints.def(
      "__bool__", [](const EqualityConstraints& self) -> bool { return self; },
      py::is_operator());

  py::class_<InequalityConstraints> inequalityConstraints{
      optimization, "InequalityConstraints"};
  inequalityConstraints.def(
      "__bool__",
      [](const InequalityConstraints& self) -> bool { return self; },
      py::is_operator());
}

}  // namespace sleipnir
