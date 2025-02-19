// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/jacobian.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_jacobian(nb::class_<Jacobian>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<VariableMatrix, VariableMatrix>(), "variables"_a, "wrt"_a,
          DOC(sleipnir, Jacobian, Jacobian));
  cls.def("get", &Jacobian::get, DOC(sleipnir, Jacobian, get));
  cls.def("value", &Jacobian::value, DOC(sleipnir, Jacobian, value));
}

}  // namespace sleipnir
