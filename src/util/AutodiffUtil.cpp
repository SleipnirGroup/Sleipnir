// Copyright (c) Sleipnir contributors

#include "util/AutodiffUtil.hpp"

#include <cassert>

namespace sleipnir {

void SetAD(Eigen::Ref<VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.rows() == src.rows());

  for (int row = 0; row < dest.rows(); ++row) {
    dest(row).SetValue(src(row));
  }
}

Eigen::VectorXd GetAD(Eigen::Ref<VectorXvar> src) {
  Eigen::VectorXd dest{src.size()};
  for (int row = 0; row < dest.size(); ++row) {
    dest(row) = src(row).Value();
  }
  return dest;
}

}  // namespace sleipnir
