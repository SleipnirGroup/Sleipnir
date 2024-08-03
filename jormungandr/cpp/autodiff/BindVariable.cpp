// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <sleipnir/autodiff/Variable.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

#if defined(__APPLE__) && defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#elif defined(__clang__) && __clang_major__ >= 7
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

namespace sleipnir {

void BindVariable(nb::module_& autodiff, nb::class_<Variable>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(sleipnir, Variable, Variable));
  cls.def(nb::init<double>(), "value"_a, DOC(sleipnir, Variable, Variable, 2));
  cls.def(
      "__init__", [](Variable* self, int value) { new (self) Variable(value); },
      "value"_a, DOC(sleipnir, Variable, Variable, 2));
  cls.def("set_value", nb::overload_cast<double>(&Variable::SetValue),
          "value"_a, DOC(sleipnir, Variable, SetValue));
  cls.def("set_value", nb::overload_cast<double>(&Variable::SetValue),
          "value"_a, DOC(sleipnir, Variable, SetValue));
  cls.def(double() * nb::self, "lhs"_a);
  cls.def(nb::self * double(), "rhs"_a);
  cls.def(nb::self * nb::self, "rhs"_a);
  cls.def(nb::self *= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, imul));
  cls.def(nb::self *= nb::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, imul));
  cls.def(double() / nb::self, "lhs"_a);
  cls.def(nb::self / double(), "rhs"_a);
  cls.def(nb::self / nb::self, "rhs"_a);
  cls.def(nb::self /= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, idiv));
  cls.def(nb::self /= nb::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, idiv));
  cls.def(double() + nb::self, "lhs"_a);
  cls.def(nb::self + double(), "rhs"_a);
  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(nb::self += double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, iadd));
  cls.def(nb::self += nb::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, iadd));
  cls.def(double() - nb::self, "lhs"_a);
  cls.def(nb::self - double(), "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(nb::self -= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, isub));
  cls.def(nb::self -= nb::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, isub));
  cls.def(
      "__pow__",
      [](const Variable& self, int power) {
        return sleipnir::pow(self, power);
      },
      nb::is_operator(), "power"_a);
  cls.def(-nb::self);
  cls.def(+nb::self);
  cls.def("value", &Variable::Value, DOC(sleipnir, Variable, Value));
  cls.def("type", &Variable::Type, DOC(sleipnir, Variable, Type));
  cls.def(nb::self == nb::self, "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self < nb::self, "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self <= nb::self, "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self > nb::self, "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self >= nb::self, "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self == double(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self < double(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self <= double(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self > double(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self >= double(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(double() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(double() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(double() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(double() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(double() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));

  autodiff.def(
      "abs", [](double x) { return sleipnir::abs(Variable{x}); }, "x"_a,
      DOC(sleipnir, abs));
  autodiff.def("abs", static_cast<Variable (*)(const Variable&)>(&abs), "x"_a,
               DOC(sleipnir, abs));
  autodiff.def(
      "acos", [](double x) { return sleipnir::acos(Variable{x}); }, "x"_a,
      DOC(sleipnir, acos));
  autodiff.def("acos", static_cast<Variable (*)(const Variable&)>(&acos), "x"_a,
               DOC(sleipnir, acos));
  autodiff.def(
      "asin", [](double x) { return sleipnir::asin(Variable{x}); }, "x"_a,
      DOC(sleipnir, asin));
  autodiff.def("asin", static_cast<Variable (*)(const Variable&)>(&asin), "x"_a,
               DOC(sleipnir, asin));
  autodiff.def(
      "atan", [](double x) { return sleipnir::atan(Variable{x}); }, "x"_a,
      DOC(sleipnir, atan));
  autodiff.def("atan", static_cast<Variable (*)(const Variable&)>(&atan), "x"_a,
               DOC(sleipnir, atan));
  autodiff.def(
      "atan2",
      [](double y, const Variable& x) { return sleipnir::atan2(y, x); }, "y"_a,
      "x"_a, DOC(sleipnir, atan2));
  autodiff.def(
      "atan2",
      [](const Variable& y, double x) { return sleipnir::atan2(y, x); }, "y"_a,
      "x"_a, DOC(sleipnir, atan2));
  autodiff.def(
      "atan2",
      [](const Variable& y, const Variable& x) {
        return sleipnir::atan2(y, x);
      },
      "y"_a, "x"_a, DOC(sleipnir, atan2));
  autodiff.def(
      "cos", [](double x) { return sleipnir::cos(Variable{x}); }, "x"_a,
      DOC(sleipnir, cos));
  autodiff.def("cos", static_cast<Variable (*)(const Variable&)>(&cos), "x"_a,
               DOC(sleipnir, cos));
  autodiff.def(
      "cosh", [](double x) { return sleipnir::cosh(Variable{x}); }, "x"_a,
      DOC(sleipnir, cosh));
  autodiff.def("cosh", static_cast<Variable (*)(const Variable&)>(&cosh), "x"_a,
               DOC(sleipnir, cosh));
  autodiff.def(
      "erf", [](double x) { return sleipnir::erf(Variable{x}); }, "x"_a,
      DOC(sleipnir, erf));
  autodiff.def("erf", static_cast<Variable (*)(const Variable&)>(&erf), "x"_a,
               DOC(sleipnir, erf));
  autodiff.def(
      "exp", [](double x) { return sleipnir::exp(Variable{x}); }, "x"_a,
      DOC(sleipnir, exp));
  autodiff.def("exp", static_cast<Variable (*)(const Variable&)>(&exp), "x"_a,
               DOC(sleipnir, exp));
  autodiff.def(
      "hypot",
      [](double x, const Variable& y) { return sleipnir::hypot(x, y); }, "x"_a,
      "y"_a, DOC(sleipnir, hypot));
  autodiff.def(
      "hypot",
      [](const Variable& x, double y) { return sleipnir::hypot(x, y); }, "x"_a,
      "y"_a, DOC(sleipnir, hypot));
  autodiff.def(
      "hypot",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&hypot),
      "x"_a, "y"_a, DOC(sleipnir, hypot));
  autodiff.def("hypot",
               static_cast<Variable (*)(const Variable&, const Variable&,
                                        const Variable&)>(&hypot),
               "x"_a, "y"_a, "z"_a, DOC(sleipnir, hypot, 2));
  autodiff.def(
      "log", [](double x) { return sleipnir::log(Variable{x}); }, "x"_a,
      DOC(sleipnir, log));
  autodiff.def("log", static_cast<Variable (*)(const Variable&)>(&log), "x"_a,
               DOC(sleipnir, log));
  autodiff.def(
      "log10", [](double x) { return sleipnir::log10(Variable{x}); }, "x"_a,
      DOC(sleipnir, log10));
  autodiff.def("log10", static_cast<Variable (*)(const Variable&)>(&log10),
               "x"_a, DOC(sleipnir, log10));
  autodiff.def(
      "pow",
      [](double base, const Variable& power) {
        return sleipnir::pow(base, power);
      },
      "base"_a, "power"_a, DOC(sleipnir, pow));
  autodiff.def(
      "pow",
      [](const Variable& base, double power) {
        return sleipnir::pow(base, power);
      },
      "base"_a, "power"_a, DOC(sleipnir, pow));
  autodiff.def(
      "pow", static_cast<Variable (*)(const Variable&, const Variable&)>(&pow),
      "base"_a, "power"_a, DOC(sleipnir, pow));
  autodiff.def(
      "sign", [](double x) { return sleipnir::sign(Variable{x}); }, "x"_a,
      DOC(sleipnir, sign));
  autodiff.def("sign", static_cast<Variable (*)(const Variable&)>(&sign), "x"_a,
               DOC(sleipnir, sign));
  autodiff.def(
      "sin", [](double x) { return sleipnir::sin(Variable{x}); }, "x"_a,
      DOC(sleipnir, sin));
  autodiff.def("sin", static_cast<Variable (*)(const Variable&)>(&sin), "x"_a,
               DOC(sleipnir, sin));
  autodiff.def(
      "sinh", [](double x) { return sleipnir::sinh(Variable{x}); }, "x"_a,
      DOC(sleipnir, sinh));
  autodiff.def("sinh", static_cast<Variable (*)(const Variable&)>(&sinh), "x"_a,
               DOC(sleipnir, sinh));
  autodiff.def(
      "sqrt", [](double x) { return sleipnir::sqrt(Variable{x}); }, "x"_a,
      DOC(sleipnir, sqrt));
  autodiff.def("sqrt", static_cast<Variable (*)(const Variable&)>(&sqrt), "x"_a,
               DOC(sleipnir, sqrt));
  autodiff.def(
      "tan", [](double x) { return sleipnir::tan(Variable{x}); }, "x"_a,
      DOC(sleipnir, tan));
  autodiff.def("tan", static_cast<Variable (*)(const Variable&)>(&tan), "x"_a,
               DOC(sleipnir, tan));
  autodiff.def(
      "tanh", [](double x) { return sleipnir::tanh(Variable{x}); }, "x"_a,
      DOC(sleipnir, tanh));
  autodiff.def("tanh", static_cast<Variable (*)(const Variable&)>(&tanh), "x"_a,
               DOC(sleipnir, tanh));
}

}  // namespace sleipnir
