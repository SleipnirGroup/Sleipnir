// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Variable.hpp"

#include "sleipnir/autodiff/ExpressionGraph.hpp"

namespace sleipnir {

double Variable::Value() {
  // Updates the value of this variable based on the values of its dependent
  // variables
  detail::ExpressionGraph{*this}.Update();

  return expr->value;
}

}  // namespace sleipnir
