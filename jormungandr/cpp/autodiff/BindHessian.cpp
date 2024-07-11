// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/Hessian.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindHessian(nb::class_<Hessian>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Hessian, Hessian));
  cls.def("get", &Hessian::Get, DOC(sleipnir, Hessian, Get));
  cls.def("value", &Hessian::Value, DOC(sleipnir, Hessian, Value));
}

}  // namespace sleipnir
