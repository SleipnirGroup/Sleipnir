// Copyright (c) Sleipnir contributors

#include "autodiff/BindExpression.hpp"

#include <sleipnir/autodiff/Expression.hpp>

namespace py = pybind11;

namespace sleipnir {

void BindExpression(py::module_& autodiff) {
  py::enum_<ExpressionType> expressionType{autodiff, "ExpressionType"};
  expressionType.value("NONE", ExpressionType::kNone)
      .value("CONSTANT", ExpressionType::kConstant)
      .value("LINEAR", ExpressionType::kLinear)
      .value("QUADRATIC", ExpressionType::kQuadratic)
      .value("NONLINEAR", ExpressionType::kNonlinear);
}

}  // namespace sleipnir
