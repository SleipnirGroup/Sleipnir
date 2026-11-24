// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>

#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/matrix.hpp>
#include <sleipnir/autodiff/real.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_jacobian(nb::class_<Jacobian<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Real<double>, Real<double>>(), "expression"_a, "wrt"_a,
          DOC(slp, Jacobian, Jacobian));
  cls.def(nb::init<Real<double>, Matrix<double>>(), "expression"_a, "wrt"_a,
          DOC(slp, Jacobian, Jacobian));
  cls.def(nb::init<Matrix<double>, Matrix<double>>(), "expressions"_a, "wrt"_a,
          DOC(slp, Jacobian, Jacobian));
  cls.def("get", &Jacobian<double>::get, DOC(slp, Jacobian, get));
  cls.def("value", &Jacobian<double>::value, DOC(slp, Jacobian, value));
}

}  // namespace slp
