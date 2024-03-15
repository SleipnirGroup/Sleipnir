// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Hessian.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindHessian(py::module_& autodiff) {
  py::class_<Hessian> cls{autodiff, "Hessian", DOC(sleipnir, Hessian)};
  cls.def(py::init<Variable, VariableMatrix>(), DOC(sleipnir, Hessian, Hessian))
      .def("get", &Hessian::Get, DOC(sleipnir, Hessian, Get))
      .def("value", &Hessian::Value, DOC(sleipnir, Hessian, Value))
      .def("update", &Hessian::Update, DOC(sleipnir, Hessian, Update));
}

}  // namespace sleipnir
