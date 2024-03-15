// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Gradient.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindGradient(py::module_& autodiff) {
  py::class_<Gradient> cls{autodiff, "Gradient", DOC(sleipnir, Gradient)};
  cls.def(py::init<Variable, Variable>(), DOC(sleipnir, Gradient, Gradient))
      .def(py::init<Variable, VariableMatrix>(),
           DOC(sleipnir, Gradient, Gradient, 2))
      .def("get", &Gradient::Get, DOC(sleipnir, Gradient, Get))
      .def(
          "value",
          [](Gradient& self) {
            return Eigen::SparseMatrix<double>{self.Value()};
          },
          DOC(sleipnir, Gradient, Value))
      .def("update", &Gradient::Update, DOC(sleipnir, Gradient, Update));
}

}  // namespace sleipnir
