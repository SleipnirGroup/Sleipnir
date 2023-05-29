// Copyright (c) Sleipnir contributors

#include "sleipnir/util/AutodiffUtil.hpp"

#include <cassert>

namespace sleipnir {

void SetAD(std::vector<Variable>& dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.size() == static_cast<size_t>(src.rows()));

  for (size_t row = 0; row < dest.size(); ++row) {
    dest[row] = src(row);
  }
}

void SetAD(Eigen::Ref<VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.rows() == src.rows());

  for (int row = 0; row < dest.rows(); ++row) {
    dest(row) = src(row);
  }
}

Eigen::VectorXd GetAD(std::vector<Variable> src) {
  Eigen::VectorXd dest{src.size()};
  for (int row = 0; row < dest.size(); ++row) {
    dest(row) = src[row].Value();
  }
  return dest;
}

Eigen::VectorXd GetAD(Eigen::Ref<VectorXvar> src) {
  Eigen::VectorXd dest{src.size()};
  for (int row = 0; row < dest.size(); ++row) {
    dest(row) = src(row).Value();
  }
  return dest;
}

}  // namespace sleipnir
