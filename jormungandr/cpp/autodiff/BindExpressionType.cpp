// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/ExpressionType.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindExpressionType(nb::enum_<ExpressionType>& e) {
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
