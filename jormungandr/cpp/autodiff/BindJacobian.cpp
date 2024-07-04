// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/Jacobian.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindJacobian(nb::class_<Jacobian>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<VariableMatrix, VariableMatrix>(), "variables"_a, "wrt"_a,
          DOC(sleipnir, Jacobian, Jacobian));
  cls.def("get", &Jacobian::Get, DOC(sleipnir, Jacobian, Get));
  cls.def("value", &Jacobian::Value, DOC(sleipnir, Jacobian, Value));
}

}  // namespace sleipnir
