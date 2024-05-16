// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/ExpressionType.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindExpressionType(py::enum_<ExpressionType>& e) {
  e.value("NONE", ExpressionType::kNone, DOC(sleipnir, ExpressionType, kNone));
  e.value("CONSTANT", ExpressionType::kConstant,
          DOC(sleipnir, ExpressionType, kConstant));
  e.value("LINEAR", ExpressionType::kLinear,
          DOC(sleipnir, ExpressionType, kLinear));
  e.value("QUADRATIC", ExpressionType::kQuadratic,
          DOC(sleipnir, ExpressionType, kQuadratic));
  e.value("NONLINEAR", ExpressionType::kNonlinear,
          DOC(sleipnir, ExpressionType, kNonlinear));
}

}  // namespace sleipnir
