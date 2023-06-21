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

void BindVariable(py::module_& autodiff, py::module_& optimization) {
  py::class_<EqualityConstraints> equalityConstraints{optimization,
                                                      "EqualityConstraints"};
  equalityConstraints.def(
      "__bool__", [](const EqualityConstraints& self) -> bool { return self; },
      py::is_operator());

  py::class_<InequalityConstraints> inequalityConstraints{
      optimization, "InequalityConstraints"};
  inequalityConstraints.def(
      "__bool__",
      [](const InequalityConstraints& self) -> bool { return self; },
      py::is_operator());

  py::class_<Variable> variable{autodiff, "Variable"};
  variable.def(py::init<>());
  variable.def(py::init<double>());
  variable.def(py::init<int>());
  variable.def("set", py::overload_cast<double>(&Variable::operator=));
  variable.def("set", py::overload_cast<int>(&Variable::operator=));
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

  autodiff.def("constant", &Constant);
  autodiff.def("abs", static_cast<Variable (*)(double)>(&abs));
  autodiff.def("abs", static_cast<Variable (*)(const Variable&)>(&abs));
  autodiff.def("acos", static_cast<Variable (*)(double)>(&acos));
  autodiff.def("acos", static_cast<Variable (*)(const Variable&)>(&acos));
  autodiff.def("asin", static_cast<Variable (*)(double)>(&asin));
  autodiff.def("asin", static_cast<Variable (*)(const Variable&)>(&asin));
  autodiff.def("atan", static_cast<Variable (*)(double)>(&atan));
  autodiff.def("atan", static_cast<Variable (*)(const Variable&)>(&atan));
  autodiff.def("atan2",
               static_cast<Variable (*)(double, const Variable&)>(&atan2));
  autodiff.def("atan2",
               static_cast<Variable (*)(const Variable&, double)>(&atan2));
  autodiff.def(
      "atan2",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&atan2));
  autodiff.def("cos", static_cast<Variable (*)(double)>(&cos));
  autodiff.def("cos", static_cast<Variable (*)(const Variable&)>(&cos));
  autodiff.def("cosh", static_cast<Variable (*)(double)>(&cosh));
  autodiff.def("cosh", static_cast<Variable (*)(const Variable&)>(&cosh));
  autodiff.def("erf", static_cast<Variable (*)(double)>(&erf));
  autodiff.def("erf", static_cast<Variable (*)(const Variable&)>(&erf));
  autodiff.def("exp", static_cast<Variable (*)(double)>(&exp));
  autodiff.def("exp", static_cast<Variable (*)(const Variable&)>(&exp));
  autodiff.def("hypot",
               static_cast<Variable (*)(double, const Variable&)>(&hypot));
  autodiff.def("hypot",
               static_cast<Variable (*)(const Variable&, double)>(&hypot));
  autodiff.def(
      "hypot",
      static_cast<Variable (*)(const Variable&, const Variable&)>(&hypot));
  autodiff.def("log", static_cast<Variable (*)(double)>(&log));
  autodiff.def("log", static_cast<Variable (*)(const Variable&)>(&log));
  autodiff.def("log10", static_cast<Variable (*)(double)>(&log10));
  autodiff.def("log10", static_cast<Variable (*)(const Variable&)>(&log10));
  autodiff.def("pow", static_cast<Variable (*)(double, const Variable&)>(&pow));
  autodiff.def("pow", static_cast<Variable (*)(const Variable&, double)>(&pow));
  autodiff.def(
      "pow", static_cast<Variable (*)(const Variable&, const Variable&)>(&pow));
  autodiff.def("sign", static_cast<Variable (*)(double)>(&sign));
  autodiff.def("sign", static_cast<Variable (*)(const Variable&)>(&sign));
  autodiff.def("sin", static_cast<Variable (*)(double)>(&sin));
  autodiff.def("sin", static_cast<Variable (*)(const Variable&)>(&sin));
  autodiff.def("sinh", static_cast<Variable (*)(double)>(&sinh));
  autodiff.def("sinh", static_cast<Variable (*)(const Variable&)>(&sinh));
  autodiff.def("sqrt", static_cast<Variable (*)(double)>(&sqrt));
  autodiff.def("sqrt", static_cast<Variable (*)(const Variable&)>(&sqrt));
  autodiff.def("tan", static_cast<Variable (*)(double)>(&tan));
  autodiff.def("tan", static_cast<Variable (*)(const Variable&)>(&tan));
  autodiff.def("tanh", static_cast<Variable (*)(double)>(&tanh));
  autodiff.def("tanh", static_cast<Variable (*)(const Variable&)>(&tanh));

  // FIXME: Eigen::SparseVector<double> isn't wrapped correctly by pybind11
  // https://github.com/pybind/pybind11/issues/2301
#if 0
  // Gradient.hpp
  {
    py::class_<Gradient> cls{autodiff, "Gradient"};
    cls.def(py::init<Variable, Variable>())
        .def(py::init<Variable, VectorXvar>())
        .def("calculate", &Gradient::Calculate)
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
#endif
}

}  // namespace sleipnir
