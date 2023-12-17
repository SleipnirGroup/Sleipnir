// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/ExpressionType.hpp"

#include <ostream>

namespace sleipnir {

void PrintTo(const ExpressionType& type, std::ostream* os) {
  using enum sleipnir::ExpressionType;

  switch (type) {
    case kNone:
      *os << "kNone";
      break;
    case kConstant:
      *os << "kConstant";
      break;
    case kLinear:
      *os << "kLinear";
      break;
    case kQuadratic:
      *os << "kQuadratic";
      break;
    case kNonlinear:
      *os << "kNonlinear";
      break;
  }
}

}  // namespace sleipnir
