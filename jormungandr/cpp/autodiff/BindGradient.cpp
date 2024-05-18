// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Gradient.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindGradient(py::class_<Gradient>& cls) {
  using namespace py::literals;

  cls.def(py::init<Variable, Variable>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient));
  cls.def(py::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Gradient, Gradient, 2));
  cls.def("get", &Gradient::Get, DOC(sleipnir, Gradient, Get));
  cls.def(
      "value",
      [](Gradient& self) { return Eigen::SparseMatrix<double>{self.Value()}; },
      DOC(sleipnir, Gradient, Value));
  cls.def("update", &Gradient::Update, DOC(sleipnir, Gradient, Update));
}

}  // namespace sleipnir
