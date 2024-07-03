// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

#if defined(__APPLE__) && defined(__clang__)
#if (__clang_major__ >= 10)
PYBIND11_WARNING_DISABLE_CLANG("-Wself-assign-overloaded")
#endif
#elif defined(__clang__)
#if (__clang_major__ >= 7)
PYBIND11_WARNING_DISABLE_CLANG("-Wself-assign-overloaded")
#endif
#endif

namespace sleipnir {

void BindVariable(py::module_& autodiff, py::class_<Variable>& cls) {
  using namespace py::literals;

  cls.def(py::init<>(), DOC(sleipnir, Variable, Variable));
  cls.def(py::init<double>(), "value"_a, DOC(sleipnir, Variable, Variable, 2));
  cls.def(py::init<int>(), "value"_a, DOC(sleipnir, Variable, Variable, 2));
  cls.def("set_value", py::overload_cast<double>(&Variable::SetValue),
          "value"_a, DOC(sleipnir, Variable, SetValue));
  cls.def("set_value", py::overload_cast<double>(&Variable::SetValue),
          "value"_a, DOC(sleipnir, Variable, SetValue));
  cls.def(double() * py::self, "lhs"_a);
  cls.def(py::self * double(), "rhs"_a);
  cls.def(py::self * py::self, "rhs"_a);
  cls.def(py::self *= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, imul));
  cls.def(py::self *= py::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, imul));
  cls.def(double() / py::self, "lhs"_a);
  cls.def(py::self / double(), "rhs"_a);
  cls.def(py::self / py::self, "rhs"_a);
  cls.def(py::self /= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, idiv));
  cls.def(py::self /= py::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, idiv));
  cls.def(double() + py::self, "lhs"_a);
  cls.def(py::self + double(), "rhs"_a);
  cls.def(py::self + py::self, "rhs"_a);
  cls.def(py::self += double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, iadd));
  cls.def(py::self += py::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, iadd));
  cls.def(double() - py::self, "lhs"_a);
  cls.def(py::self - double(), "rhs"_a);
  cls.def(py::self - py::self, "rhs"_a);
  cls.def(py::self -= double(), "rhs"_a,
          DOC(sleipnir, Variable, operator, isub));
  cls.def(py::self -= py::self, "rhs"_a,
          DOC(sleipnir, Variable, operator, isub));
  cls.def(
      "__pow__",
      [](const Variable& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator(), "power"_a);
  cls.def(-py::self);
  cls.def(+py::self);
  cls.def("value", &Variable::Value, DOC(sleipnir, Variable, Value));
  cls.def("type", &Variable::Type, DOC(sleipnir, Variable, Type));
  cls.def(py::self == py::self, "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self < py::self, "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self <= py::self, "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self > py::self, "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self >= py::self, "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(py::self == double(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(py::self < double(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(py::self <= double(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(py::self > double(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(py::self >= double(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(double() == py::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(double() < py::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(double() <= py::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(double() > py::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(double() >= py::self, "lhs"_a, DOC(sleipnir, operator, ge));

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
