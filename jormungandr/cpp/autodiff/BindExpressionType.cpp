// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/ExpressionType.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindExpressionType(py::module_& autodiff) {
  py::enum_<ExpressionType> expressionType{autodiff, "ExpressionType",
                                           DOC(sleipnir, ExpressionType)};
  expressionType
      .value("NONE", ExpressionType::kNone,
             DOC(sleipnir, ExpressionType, kNone))
      .value("CONSTANT", ExpressionType::kConstant,
             DOC(sleipnir, ExpressionType, kConstant))
      .value("LINEAR", ExpressionType::kLinear,
             DOC(sleipnir, ExpressionType, kLinear))
      .value("QUADRATIC", ExpressionType::kQuadratic,
             DOC(sleipnir, ExpressionType, kQuadratic))
      .value("NONLINEAR", ExpressionType::kNonlinear,
             DOC(sleipnir, ExpressionType, kNonlinear));
}

}  // namespace sleipnir
