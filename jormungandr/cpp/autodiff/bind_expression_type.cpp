// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/expression_type.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_expression_type(nb::enum_<ExpressionType>& e) {
  e.value("NONE", ExpressionType::NONE, DOC(sleipnir, ExpressionType, NONE));
  e.value("CONSTANT", ExpressionType::CONSTANT,
          DOC(sleipnir, ExpressionType, CONSTANT));
  e.value("LINEAR", ExpressionType::LINEAR,
          DOC(sleipnir, ExpressionType, LINEAR));
  e.value("QUADRATIC", ExpressionType::QUADRATIC,
          DOC(sleipnir, ExpressionType, QUADRATIC));
  e.value("NONLINEAR", ExpressionType::NONLINEAR,
          DOC(sleipnir, ExpressionType, NONLINEAR));
}

}  // namespace sleipnir
