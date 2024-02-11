// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/ExpressionType.hpp"

#include <ostream>

namespace sleipnir {

std::ostream& operator<<(std::ostream& os, const ExpressionType& type) {
  using enum sleipnir::ExpressionType;

  switch (type) {
    case kNone:
      os << "kNone";
      break;
    case kConstant:
      os << "kConstant";
      break;
    case kLinear:
      os << "kLinear";
      break;
    case kQuadratic:
      os << "kQuadratic";
      break;
    case kNonlinear:
      os << "kNonlinear";
      break;
  }

  return os;
}

}  // namespace sleipnir
