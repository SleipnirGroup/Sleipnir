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

void BindVariable(py::module_& autodiff) {
  py::class_<Variable> variable{autodiff, "Variable", DOC(sleipnir, Variable)};
  variable.def(py::init<>(), DOC(sleipnir, Variable, Variable));
  variable.def(py::init<double>(), DOC(sleipnir, Variable, Variable, 2));
  variable.def(py::init<int>(), DOC(sleipnir, Variable, Variable, 3));
  variable.def("set", py::overload_cast<double>(&Variable::operator=),
               DOC(sleipnir, Variable, operator, assign));
  variable.def("set", py::overload_cast<int>(&Variable::operator=),
               DOC(sleipnir, Variable, operator, assign, 2));
  variable.def("set_value", py::overload_cast<double>(&Variable::SetValue),
               DOC(sleipnir, Variable, SetValue));
  variable.def("set_value", py::overload_cast<int>(&Variable::SetValue),
               DOC(sleipnir, Variable, SetValue, 2));
  variable.def(double() * py::self);
  variable.def(py::self * double());
  variable.def(py::self * py::self);
  variable.def(py::self *= double(), DOC(sleipnir, Variable, operator, imul));
  variable.def(py::self *= py::self, DOC(sleipnir, Variable, operator, imul));
  variable.def(double() / py::self);
  variable.def(py::self / double());
  variable.def(py::self / py::self);
  variable.def(py::self /= double(), DOC(sleipnir, Variable, operator, idiv));
  variable.def(py::self /= py::self, DOC(sleipnir, Variable, operator, idiv));
  variable.def(double() + py::self);
  variable.def(py::self + double());
  variable.def(py::self + py::self);
  variable.def(py::self += double(), DOC(sleipnir, Variable, operator, iadd));
  variable.def(py::self += py::self, DOC(sleipnir, Variable, operator, iadd));
  variable.def(double() - py::self);
  variable.def(py::self - double());
  variable.def(py::self - py::self);
  variable.def(py::self -= double(), DOC(sleipnir, Variable, operator, isub));
  variable.def(py::self -= py::self, DOC(sleipnir, Variable, operator, isub));
  variable.def(
      "__pow__",
      [](const Variable& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator());
  variable.def(-py::self);
  variable.def(+py::self);
  variable.def("value", &Variable::Value, DOC(sleipnir, Variable, Value));
  variable.def("type", &Variable::Type, DOC(sleipnir, Variable, Type));
  variable.def("update", &Variable::Update, DOC(sleipnir, Variable, Update));
  variable.def(py::self == py::self, DOC(sleipnir, operator, eq));
  variable.def(py::self < py::self, DOC(sleipnir, operator, lt));
  variable.def(py::self <= py::self, DOC(sleipnir, operator, le));
  variable.def(py::self > py::self, DOC(sleipnir, operator, gt));
  variable.def(py::self >= py::self, DOC(sleipnir, operator, ge));
  variable.def(py::self == double(), DOC(sleipnir, operator, eq));
  variable.def(py::self < double(), DOC(sleipnir, operator, lt));
  variable.def(py::self <= double(), DOC(sleipnir, operator, le));
  variable.def(py::self > double(), DOC(sleipnir, operator, gt));
  variable.def(py::self >= double(), DOC(sleipnir, operator, ge));
  variable.def(double() == py::self, DOC(sleipnir, operator, eq));
  variable.def(double() < py::self, DOC(sleipnir, operator, lt));
  variable.def(double() <= py::self, DOC(sleipnir, operator, le));
  variable.def(double() > py::self, DOC(sleipnir, operator, gt));
  variable.def(double() >= py::self, DOC(sleipnir, operator, ge));

  autodiff.def(
      "abs", [](double x) { return sleipnir::abs(Variable{x}); },
      DOC(sleipnir, abs));
  autodiff.def("abs", static_cast<Variable (*)(const Variable&)>(&abs),
               DOC(sleipnir, abs));
  autodiff.def(
      "acos", [](double x) { return sleipnir::acos(Variable{x}); },
      DOC(sleipnir, acos));
  autodiff.def("acos", static_cast<Variable (*)(const Variable&)>(&acos),
               DOC(sleipnir, acos));
  autodiff.def(
      "asin", [](double x) { return sleipnir::asin(Variable{x}); },
      DOC(sleipnir, asin));
  autodiff.def("asin", static_cast<Variable (*)(const Variable&)>(&asin),
               DOC(sleipnir, asin));
  autodiff.def(
      "atan", [](double x) { return sleipnir::atan(Variable{x}); },
      DOC(sleipnir, atan));
  autodiff.def("atan", static_cast<Variable (*)(const Variable&)>(&atan),
               DOC(sleipnir, atan));
  autodiff.def(
      "atan2",
      [](double y, const Variable& x) { return sleipnir::atan2(y, x); },
      DOC(sleipnir, atan2));
  autodiff.def(
      "atan2",
      [](const Variable& y, double x) { return sleipnir::atan2(y, x); },
      DOC(sleipnir, atan2));
  autodiff.def(
      "atan2",
      [](const Variable& y, const Variable& x) {
        return sleipnir::atan2(y, x);
      },
      DOC(sleipnir, atan2));
  autodiff.def(
      "cos", [](double x) { return sleipnir::cos(Variable{x}); },
      DOC(sleipnir, cos));
  autodiff.def("cos", static_cast<Variable (*)(const Variable&)>(&cos),
               DOC(sleipnir, cos));
  autodiff.def(
      "cosh", [](double x) { return sleipnir::cosh(Variable{x}); },
      DOC(sleipnir, cosh));
  autodiff.def("cosh", static_cast<Variable (*)(const Variable&)>(&cosh),
               DOC(sleipnir, cosh));
  autodiff.def(
      "erf", [](double x) { return sleipnir::erf(Variable{x}); },
      DOC(sleipnir, erf));
  autodiff.def("erf", static_cast<Variable (*)(const Variable&)>(&erf),
               DOC(sleipnir, erf));
  autodiff.def(
      "exp", [](double x) { return sleipnir::exp(Variable{x}); },
      DOC(sleipnir, exp));
  autodiff.def("exp", static_cast<Variable (*)(const Variable&)>(&exp),
               DOC(sleipnir, exp));
  autodiff.def(
      "hypot",
      [](double x, const Variable& y) { return sleipnir::hypot(x, y); },
      DOC(sleipnir, hypot));
  autodiff.def(
      "hypot",
      [](const Variable& x, double y) { return sleipnir::hypot(x, y); },
      DOC(sleipnir, hypot));
  autodiff.def(
      "hypot",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&hypot),
      DOC(sleipnir, hypot));
  autodiff.def("hypot",
               static_cast<Variable (*)(const Variable&, const Variable&,
                                        const Variable&)>(&hypot),
               DOC(sleipnir, hypot, 2));
  autodiff.def(
      "log", [](double x) { return sleipnir::log(Variable{x}); },
      DOC(sleipnir, log));
  autodiff.def("log", static_cast<Variable (*)(const Variable&)>(&log),
               DOC(sleipnir, log));
  autodiff.def(
      "log10", [](double x) { return sleipnir::log10(Variable{x}); },
      DOC(sleipnir, log10));
  autodiff.def("log10", static_cast<Variable (*)(const Variable&)>(&log10),
               DOC(sleipnir, log10));
  autodiff.def(
      "pow",
      [](double base, const Variable& power) {
        return sleipnir::pow(base, power);
      },
      DOC(sleipnir, pow));
  autodiff.def(
      "pow",
      [](const Variable& base, double power) {
        return sleipnir::pow(base, power);
      },
      DOC(sleipnir, pow));
  autodiff.def(
      "pow", static_cast<Variable (*)(const Variable&, const Variable&)>(&pow),
      DOC(sleipnir, pow));
  autodiff.def(
      "sign", [](double x) { return sleipnir::sign(Variable{x}); },
      DOC(sleipnir, sign));
  autodiff.def("sign", static_cast<Variable (*)(const Variable&)>(&sign),
               DOC(sleipnir, sign));
  autodiff.def(
      "sin", [](double x) { return sleipnir::sin(Variable{x}); },
      DOC(sleipnir, sin));
  autodiff.def("sin", static_cast<Variable (*)(const Variable&)>(&sin),
               DOC(sleipnir, sin));
  autodiff.def(
      "sinh", [](double x) { return sleipnir::sinh(Variable{x}); },
      DOC(sleipnir, sinh));
  autodiff.def("sinh", static_cast<Variable (*)(const Variable&)>(&sinh),
               DOC(sleipnir, sinh));
  autodiff.def(
      "sqrt", [](double x) { return sleipnir::sqrt(Variable{x}); },
      DOC(sleipnir, sqrt));
  autodiff.def("sqrt", static_cast<Variable (*)(const Variable&)>(&sqrt),
               DOC(sleipnir, sqrt));
  autodiff.def(
      "tan", [](double x) { return sleipnir::tan(Variable{x}); },
      DOC(sleipnir, tan));
  autodiff.def("tan", static_cast<Variable (*)(const Variable&)>(&tan),
               DOC(sleipnir, tan));
  autodiff.def(
      "tanh", [](double x) { return sleipnir::tanh(Variable{x}); },
      DOC(sleipnir, tanh));
  autodiff.def("tanh", static_cast<Variable (*)(const Variable&)>(&tanh),
               DOC(sleipnir, tanh));
}

}  // namespace sleipnir
