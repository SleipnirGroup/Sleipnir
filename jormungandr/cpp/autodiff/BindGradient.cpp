// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/Gradient.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindGradient(nb::class_<Gradient>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable, Variable>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient));
  cls.def(nb::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient, 2));
  cls.def("get", &Gradient::Get, DOC(sleipnir, Gradient, Get));
  cls.def(
      "value",
      [](Gradient& self) { return Eigen::SparseMatrix<double>{self.Value()}; },
      DOC(sleipnir, Gradient, Value));
}

}  // namespace sleipnir
