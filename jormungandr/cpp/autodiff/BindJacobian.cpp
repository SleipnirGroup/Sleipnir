// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Jacobian.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindJacobian(py::class_<Jacobian>& cls) {
  using namespace py::literals;

  cls.def(py::init<VariableMatrix, VariableMatrix>(), "variables"_a, "wrt"_a,
          DOC(sleipnir, Jacobian, Jacobian));
  cls.def("get", &Jacobian::Get, DOC(sleipnir, Jacobian, Get));
  cls.def("value", &Jacobian::Value, DOC(sleipnir, Jacobian, Value));
  cls.def("update", &Jacobian::Update, DOC(sleipnir, Jacobian, Update));
}

}  // namespace sleipnir
