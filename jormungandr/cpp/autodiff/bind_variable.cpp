// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <sleipnir/autodiff/variable.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

#if defined(__APPLE__) && defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#elif defined(__clang__) && __clang_major__ >= 7
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

namespace slp {

void bind_variable(nb::module_& autodiff, nb::class_<Variable>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(slp, Variable, Variable));
  cls.def(nb::init<double>(), "value"_a, DOC(slp, Variable, Variable, 2));
  cls.def(
      "__init__", [](Variable* self, int value) { new (self) Variable(value); },
      "value"_a, DOC(slp, Variable, Variable, 2));
  cls.def("set_value", nb::overload_cast<double>(&Variable::set_value),
          "value"_a, DOC(slp, Variable, set_value));
  cls.def("set_value", nb::overload_cast<double>(&Variable::set_value),
          "value"_a, DOC(slp, Variable, set_value));
  cls.def(double() * nb::self, "lhs"_a);
  cls.def(nb::self * double(), "rhs"_a);
  cls.def(nb::self * nb::self, "rhs"_a);
  cls.def(nb::self *= double(), "rhs"_a, DOC(slp, Variable, operator, imul));
  cls.def(nb::self *= nb::self, "rhs"_a, DOC(slp, Variable, operator, imul));
  cls.def(double() / nb::self, "lhs"_a);
  cls.def(nb::self / double(), "rhs"_a);
  cls.def(nb::self / nb::self, "rhs"_a);
  cls.def(nb::self /= double(), "rhs"_a, DOC(slp, Variable, operator, idiv));
  cls.def(nb::self /= nb::self, "rhs"_a, DOC(slp, Variable, operator, idiv));
  cls.def(double() + nb::self, "lhs"_a);
  cls.def(nb::self + double(), "rhs"_a);
  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(nb::self += double(), "rhs"_a, DOC(slp, Variable, operator, iadd));
  cls.def(nb::self += nb::self, "rhs"_a, DOC(slp, Variable, operator, iadd));
  cls.def(double() - nb::self, "lhs"_a);
  cls.def(nb::self - double(), "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(nb::self -= double(), "rhs"_a, DOC(slp, Variable, operator, isub));
  cls.def(nb::self -= nb::self, "rhs"_a, DOC(slp, Variable, operator, isub));
  cls.def(
      "__pow__",
      [](const Variable& self, int power) { return slp::pow(self, power); },
      nb::is_operator(), "power"_a);
  cls.def(-nb::self);
  cls.def(+nb::self);
  cls.def("value", &Variable::value, DOC(slp, Variable, value));
  cls.def("type", &Variable::type, DOC(slp, Variable, type));
  cls.def(nb::self == nb::self, "rhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self < nb::self, "rhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self <= nb::self, "rhs"_a, DOC(slp, operator, le));
  cls.def(nb::self > nb::self, "rhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self >= nb::self, "rhs"_a, DOC(slp, operator, ge));
  cls.def(nb::self == double(), "rhs"_a, DOC(slp, operator, eq));
  cls.def(nb::self < double(), "rhs"_a, DOC(slp, operator, lt));
  cls.def(nb::self <= double(), "rhs"_a, DOC(slp, operator, le));
  cls.def(nb::self > double(), "rhs"_a, DOC(slp, operator, gt));
  cls.def(nb::self >= double(), "rhs"_a, DOC(slp, operator, ge));
  cls.def(double() == nb::self, "lhs"_a, DOC(slp, operator, eq));
  cls.def(double() < nb::self, "lhs"_a, DOC(slp, operator, lt));
  cls.def(double() <= nb::self, "lhs"_a, DOC(slp, operator, le));
  cls.def(double() > nb::self, "lhs"_a, DOC(slp, operator, gt));
  cls.def(double() >= nb::self, "lhs"_a, DOC(slp, operator, ge));

  autodiff.def(
      "abs", [](double x) { return slp::abs(Variable{x}); }, "x"_a,
      DOC(slp, abs));
  autodiff.def("abs", static_cast<Variable (*)(const Variable&)>(&abs), "x"_a,
               DOC(slp, abs));
  autodiff.def(
      "acos", [](double x) { return slp::acos(Variable{x}); }, "x"_a,
      DOC(slp, acos));
  autodiff.def("acos", static_cast<Variable (*)(const Variable&)>(&acos), "x"_a,
               DOC(slp, acos));
  autodiff.def(
      "asin", [](double x) { return slp::asin(Variable{x}); }, "x"_a,
      DOC(slp, asin));
  autodiff.def("asin", static_cast<Variable (*)(const Variable&)>(&asin), "x"_a,
               DOC(slp, asin));
  autodiff.def(
      "atan", [](double x) { return slp::atan(Variable{x}); }, "x"_a,
      DOC(slp, atan));
  autodiff.def("atan", static_cast<Variable (*)(const Variable&)>(&atan), "x"_a,
               DOC(slp, atan));
  autodiff.def(
      "atan2", [](double y, const Variable& x) { return slp::atan2(y, x); },
      "y"_a, "x"_a, DOC(slp, atan2));
  autodiff.def(
      "atan2", [](const Variable& y, double x) { return slp::atan2(y, x); },
      "y"_a, "x"_a, DOC(slp, atan2));
  autodiff.def(
      "atan2",
      [](const Variable& y, const Variable& x) { return slp::atan2(y, x); },
      "y"_a, "x"_a, DOC(slp, atan2));
  autodiff.def(
      "cos", [](double x) { return slp::cos(Variable{x}); }, "x"_a,
      DOC(slp, cos));
  autodiff.def("cos", static_cast<Variable (*)(const Variable&)>(&cos), "x"_a,
               DOC(slp, cos));
  autodiff.def(
      "cosh", [](double x) { return slp::cosh(Variable{x}); }, "x"_a,
      DOC(slp, cosh));
  autodiff.def("cosh", static_cast<Variable (*)(const Variable&)>(&cosh), "x"_a,
               DOC(slp, cosh));
  autodiff.def(
      "erf", [](double x) { return slp::erf(Variable{x}); }, "x"_a,
      DOC(slp, erf));
  autodiff.def("erf", static_cast<Variable (*)(const Variable&)>(&erf), "x"_a,
               DOC(slp, erf));
  autodiff.def(
      "exp", [](double x) { return slp::exp(Variable{x}); }, "x"_a,
      DOC(slp, exp));
  autodiff.def("exp", static_cast<Variable (*)(const Variable&)>(&exp), "x"_a,
               DOC(slp, exp));
  autodiff.def(
      "hypot", [](double x, const Variable& y) { return slp::hypot(x, y); },
      "x"_a, "y"_a, DOC(slp, hypot));
  autodiff.def(
      "hypot", [](const Variable& x, double y) { return slp::hypot(x, y); },
      "x"_a, "y"_a, DOC(slp, hypot));
  autodiff.def(
      "hypot",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&hypot),
      "x"_a, "y"_a, DOC(slp, hypot));
  autodiff.def("hypot",
               static_cast<Variable (*)(const Variable&, const Variable&,
                                        const Variable&)>(&hypot),
               "x"_a, "y"_a, "z"_a, DOC(slp, hypot, 2));
  autodiff.def(
      "log", [](double x) { return slp::log(Variable{x}); }, "x"_a,
      DOC(slp, log));
  autodiff.def("log", static_cast<Variable (*)(const Variable&)>(&log), "x"_a,
               DOC(slp, log));
  autodiff.def(
      "log10", [](double x) { return slp::log10(Variable{x}); }, "x"_a,
      DOC(slp, log10));
  autodiff.def("log10", static_cast<Variable (*)(const Variable&)>(&log10),
               "x"_a, DOC(slp, log10));
  autodiff.def(
      "pow",
      [](double base, const Variable& power) { return slp::pow(base, power); },
      "base"_a, "power"_a, DOC(slp, pow));
  autodiff.def(
      "pow",
      [](const Variable& base, double power) { return slp::pow(base, power); },
      "base"_a, "power"_a, DOC(slp, pow));
  autodiff.def(
      "pow", static_cast<Variable (*)(const Variable&, const Variable&)>(&pow),
      "base"_a, "power"_a, DOC(slp, pow));
  autodiff.def(
      "sign", [](double x) { return slp::sign(Variable{x}); }, "x"_a,
      DOC(slp, sign));
  autodiff.def("sign", static_cast<Variable (*)(const Variable&)>(&sign), "x"_a,
               DOC(slp, sign));
  autodiff.def(
      "sin", [](double x) { return slp::sin(Variable{x}); }, "x"_a,
      DOC(slp, sin));
  autodiff.def("sin", static_cast<Variable (*)(const Variable&)>(&sin), "x"_a,
               DOC(slp, sin));
  autodiff.def(
      "sinh", [](double x) { return slp::sinh(Variable{x}); }, "x"_a,
      DOC(slp, sinh));
  autodiff.def("sinh", static_cast<Variable (*)(const Variable&)>(&sinh), "x"_a,
               DOC(slp, sinh));
  autodiff.def(
      "sqrt", [](double x) { return slp::sqrt(Variable{x}); }, "x"_a,
      DOC(slp, sqrt));
  autodiff.def("sqrt", static_cast<Variable (*)(const Variable&)>(&sqrt), "x"_a,
               DOC(slp, sqrt));
  autodiff.def(
      "tan", [](double x) { return slp::tan(Variable{x}); }, "x"_a,
      DOC(slp, tan));
  autodiff.def("tan", static_cast<Variable (*)(const Variable&)>(&tan), "x"_a,
               DOC(slp, tan));
  autodiff.def(
      "tanh", [](double x) { return slp::tanh(Variable{x}); }, "x"_a,
      DOC(slp, tanh));
  autodiff.def("tanh", static_cast<Variable (*)(const Variable&)>(&tanh), "x"_a,
               DOC(slp, tanh));
}

}  // namespace slp
