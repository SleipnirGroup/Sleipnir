// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Jacobian.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindJacobian(py::module_& autodiff) {
  py::class_<Jacobian> cls{autodiff, "Jacobian", DOC(sleipnir, Jacobian)};
  cls.def(py::init<VariableMatrix, VariableMatrix>(),
          DOC(sleipnir, Jacobian, Jacobian))
      .def("get", &Jacobian::Get, DOC(sleipnir, Jacobian, Get))
      .def("value", &Jacobian::Value, DOC(sleipnir, Jacobian, Value))
      .def("update", &Jacobian::Update, DOC(sleipnir, Jacobian, Update));
}

}  // namespace sleipnir
