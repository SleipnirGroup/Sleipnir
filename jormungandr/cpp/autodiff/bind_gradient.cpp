// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/gradient.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_gradient(nb::class_<Gradient>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable, Variable>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient));
  cls.def(nb::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient, 2));
  cls.def("get", &Gradient::get, DOC(sleipnir, Gradient, get));
  cls.def(
      "value",
      [](Gradient& self) { return Eigen::SparseMatrix<double>{self.value()}; },
      DOC(sleipnir, Gradient, value));
}

}  // namespace sleipnir
