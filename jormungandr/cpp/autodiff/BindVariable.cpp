// Copyright (c) Sleipnir contributors

#include "autodiff/BindVariable.hpp"

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>
#include <sleipnir/autodiff/Jacobian.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/optimization/Constraints.hpp>

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

namespace pybind11::detail {
template <>
class type_caster<sleipnir::VectorXvar, void>
    : public type_caster_base<sleipnir::VectorXvar> {};
}  // namespace pybind11::detail

namespace sleipnir {

void BindVariable(py::module_& autodiff) {
  py::class_<Variable> variable{autodiff, "Variable"};
  variable.def(py::init<>());
  variable.def(py::init<double>());
  variable.def(py::init<int>());
  variable.def("set", py::overload_cast<double>(&Variable::operator=));
  variable.def("set", py::overload_cast<int>(&Variable::operator=));
  variable.def("set_value", py::overload_cast<double>(&Variable::SetValue));
  variable.def("set_value", py::overload_cast<int>(&Variable::SetValue));
  variable.def(double() * py::self);
  variable.def(py::self * double());
  variable.def(py::self * py::self);
  variable.def(py::self *= double());
  variable.def(py::self *= py::self);
  variable.def(double() / py::self);
  variable.def(py::self / double());
  variable.def(py::self / py::self);
  variable.def(py::self /= double());
  variable.def(py::self /= py::self);
  variable.def(double() + py::self);
  variable.def(py::self + double());
  variable.def(py::self + py::self);
  variable.def(py::self += double());
  variable.def(py::self += py::self);
  variable.def(double() - py::self);
  variable.def(py::self - double());
  variable.def(py::self - py::self);
  variable.def(py::self -= double());
  variable.def(py::self -= py::self);
  variable.def(
      "__pow__",
      [](const Variable& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator());
  variable.def(-py::self);
  variable.def(+py::self);
  variable.def("value", &Variable::Value);
  variable.def("update", &Variable::Update);
  variable.def(py::self == py::self);
  variable.def(py::self < py::self);
  variable.def(py::self <= py::self);
  variable.def(py::self > py::self);
  variable.def(py::self >= py::self);
  variable.def(py::self == double());
  variable.def(py::self < double());
  variable.def(py::self <= double());
  variable.def(py::self > double());
  variable.def(py::self >= double());
  variable.def(double() == py::self);
  variable.def(double() < py::self);
  variable.def(double() <= py::self);
  variable.def(double() > py::self);
  variable.def(double() >= py::self);

  autodiff.def("abs", [](double x) { return sleipnir::abs(Variable{x}); });
  autodiff.def("abs", static_cast<Variable (*)(const Variable&)>(&abs));
  autodiff.def("acos", [](double x) { return sleipnir::acos(Variable{x}); });
  autodiff.def("acos", static_cast<Variable (*)(const Variable&)>(&acos));
  autodiff.def("asin", [](double x) { return sleipnir::asin(Variable{x}); });
  autodiff.def("asin", static_cast<Variable (*)(const Variable&)>(&asin));
  autodiff.def("atan", [](double x) { return sleipnir::atan(Variable{x}); });
  autodiff.def("atan", static_cast<Variable (*)(const Variable&)>(&atan));
  autodiff.def("atan2", [](double y, const Variable& x) {
    return sleipnir::atan2(y, x);
  });
  autodiff.def("atan2", [](const Variable& y, double x) {
    return sleipnir::atan2(y, x);
  });
  autodiff.def("atan2", [](const Variable& y, const Variable& x) {
    return sleipnir::atan2(y, x);
  });
  autodiff.def("cos", [](double x) { return sleipnir::cos(Variable{x}); });
  autodiff.def("cos", static_cast<Variable (*)(const Variable&)>(&cos));
  autodiff.def("cosh", [](double x) { return sleipnir::cosh(Variable{x}); });
  autodiff.def("cosh", static_cast<Variable (*)(const Variable&)>(&cosh));
  autodiff.def("erf", [](double x) { return sleipnir::erf(Variable{x}); });
  autodiff.def("erf", static_cast<Variable (*)(const Variable&)>(&erf));
  autodiff.def("exp", [](double x) { return sleipnir::exp(Variable{x}); });
  autodiff.def("exp", static_cast<Variable (*)(const Variable&)>(&exp));
  autodiff.def("hypot", [](double x, const Variable& y) {
    return sleipnir::hypot(x, y);
  });
  autodiff.def("hypot", [](const Variable& x, double y) {
    return sleipnir::hypot(x, y);
  });
  autodiff.def(
      "hypot",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&hypot));
  autodiff.def("hypot",
               static_cast<Variable (*)(const Variable&, const Variable&,
                                        const Variable&)>(&hypot));
  autodiff.def("log", [](double x) { return sleipnir::log(Variable{x}); });
  autodiff.def("log", static_cast<Variable (*)(const Variable&)>(&log));
  autodiff.def("log10", [](double x) { return sleipnir::log10(Variable{x}); });
  autodiff.def("log10", static_cast<Variable (*)(const Variable&)>(&log10));
  autodiff.def("pow", [](double base, const Variable& power) {
    return sleipnir::pow(base, power);
  });
  autodiff.def("pow", [](const Variable& base, double power) {
    return sleipnir::pow(base, power);
  });
  autodiff.def(
      "pow", static_cast<Variable (*)(const Variable&, const Variable&)>(&pow));
  autodiff.def("sign", [](double x) { return sleipnir::sign(Variable{x}); });
  autodiff.def("sign", static_cast<Variable (*)(const Variable&)>(&sign));
  autodiff.def("sin", [](double x) { return sleipnir::sin(Variable{x}); });
  autodiff.def("sin", static_cast<Variable (*)(const Variable&)>(&sin));
  autodiff.def("sinh", [](double x) { return sleipnir::sinh(Variable{x}); });
  autodiff.def("sinh", static_cast<Variable (*)(const Variable&)>(&sinh));
  autodiff.def("sqrt", [](double x) { return sleipnir::sqrt(Variable{x}); });
  autodiff.def("sqrt", static_cast<Variable (*)(const Variable&)>(&sqrt));
  autodiff.def("tan", [](double x) { return sleipnir::tan(Variable{x}); });
  autodiff.def("tan", static_cast<Variable (*)(const Variable&)>(&tan));
  autodiff.def("tanh", [](double x) { return sleipnir::tanh(Variable{x}); });
  autodiff.def("tanh", static_cast<Variable (*)(const Variable&)>(&tanh));

  // Gradient.hpp
  {
    py::class_<Gradient> cls{autodiff, "Gradient"};
    cls.def(py::init<Variable, Variable>())
        .def(py::init<Variable, VectorXvar>())
        .def("calculate",
             [](Gradient& self) {
               return Eigen::SparseMatrix<double>{self.Calculate()};
             })
        .def("update", &Gradient::Update);
  }

  // Jacobian.hpp
  {
    py::class_<Jacobian> cls{autodiff, "Jacobian"};
    cls.def(py::init<VectorXvar, VectorXvar>())
        .def("calculate", &Jacobian::Calculate)
        .def("update", &Jacobian::Update);
  }

  // Hessian.hpp
  {
    py::class_<Hessian> cls{autodiff, "Hessian"};
    cls.def(py::init<Variable, VectorXvar>())
        .def("calculate", &Hessian::Calculate)
        .def("update", &Hessian::Update);
  }
}

}  // namespace sleipnir
