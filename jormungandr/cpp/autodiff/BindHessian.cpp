// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Hessian.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindHessian(py::class_<Hessian>& cls) {
  using namespace py::literals;

  cls.def(py::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Hessian, Hessian));
  cls.def("get", &Hessian::Get, DOC(sleipnir, Hessian, Get));
  cls.def("value", &Hessian::Value, DOC(sleipnir, Hessian, Value));
  cls.def("update", &Hessian::Update, DOC(sleipnir, Hessian, Update));
}

}  // namespace sleipnir
