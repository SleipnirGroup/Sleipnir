// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/hessian.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_hessian(nb::class_<Hessian<>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable, VariableMatrix>(), "variable"_a, "wrt"_a,
          DOC(sleipnir, Hessian, Hessian));
  cls.def("get", &Hessian<>::get, DOC(sleipnir, Hessian, get));
  cls.def("value", &Hessian<>::value, DOC(sleipnir, Hessian, value));
}

}  // namespace sleipnir
