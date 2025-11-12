// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/expression_type.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_expression_type(nb::enum_<ExpressionType>& e) {
  e.value("NONE", ExpressionType::NONE, DOC(slp, ExpressionType, NONE));
  e.value("CONSTANT", ExpressionType::CONSTANT,
          DOC(slp, ExpressionType, CONSTANT));
  e.value("LINEAR", ExpressionType::LINEAR, DOC(slp, ExpressionType, LINEAR));
  e.value("QUADRATIC", ExpressionType::QUADRATIC,
          DOC(slp, ExpressionType, QUADRATIC));
  e.value("NONLINEAR", ExpressionType::NONLINEAR,
          DOC(slp, ExpressionType, NONLINEAR));
}

}  // namespace slp
