// Copyright (c) Sleipnir contributors

#include "DifferentialDriveUtil.hpp"

#include <units/acceleration.h>
#include <units/length.h>
#include <units/velocity.h>
#include <units/voltage.h>

Eigen::Vector<double, 5> DifferentialDriveDynamicsDouble(
    const Eigen::Vector<double, 5>& x, const Eigen::Vector<double, 2>& u) {
  // x = [x, y, heading, left velocity, right velocity]ᵀ
  // u = [left voltage, right voltage]ᵀ
  constexpr double trackwidth = (0.699_m).value();
  constexpr double Kv_linear = (3.02_V / 1_mps).value();
  constexpr double Ka_linear = (0.642_V / 1_mps_sq).value();
  constexpr double Kv_angular = (1.382_V / 1_mps).value();
  constexpr double Ka_angular = (0.08495_V / 1_mps_sq).value();

  double v = (x(3) + x(4)) / 2.0;

  constexpr double A1 =
      -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
  constexpr double A2 =
      -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
  constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
  constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
  Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
  Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

  Eigen::Vector<double, 5> xdot;
  xdot(0) = v * std::cos(x(2));
  xdot(1) = v * std::sin(x(2));
  xdot(2) = (x(4) - x(3)) / trackwidth;
  xdot.segment(3, 2) = A * x.block<2, 1>(3, 0) + B * u;
  return xdot;
}

sleipnir::VariableMatrix DifferentialDriveDynamics(
    const sleipnir::VariableMatrix& x, const sleipnir::VariableMatrix& u) {
  // x = [x, y, heading, left velocity, right velocity]ᵀ
  // u = [left voltage, right voltage]ᵀ
  constexpr double trackwidth = (0.699_m).value();
  constexpr double Kv_linear = (3.02_V / 1_mps).value();
  constexpr double Ka_linear = (0.642_V / 1_mps_sq).value();
  constexpr double Kv_angular = (1.382_V / 1_mps).value();
  constexpr double Ka_angular = (0.08495_V / 1_mps_sq).value();

  auto v = (x(3) + x(4)) / 2.0;

  constexpr double A1 =
      -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
  constexpr double A2 =
      -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
  constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
  constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
  Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
  Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

  sleipnir::VariableMatrix xdot{5};
  xdot(0) = v * sleipnir::cos(x(2));
  xdot(1) = v * sleipnir::sin(x(2));
  xdot(2) = (x(4) - x(3)) / trackwidth;
  xdot.Segment(3, 2) = A * x.Segment(3, 2) + B * u;
  return xdot;
}
