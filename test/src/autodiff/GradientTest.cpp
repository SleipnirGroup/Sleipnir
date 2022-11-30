// Copyright (c) Joshua Nichols and Tyler Veness

#include <numbers>

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Gradient.hpp>

TEST(GradientTest, TrivialCase) {
  sleipnir::Variable a = 10;
  sleipnir::Variable b = 20;
  sleipnir::Variable c = a;

  EXPECT_DOUBLE_EQ(1, sleipnir::Gradient(a, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(0, sleipnir::Gradient(a, b).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1, sleipnir::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(0, sleipnir::Gradient(c, b).Calculate().coeff(0));
}

TEST(GradientTest, PositiveOperator) {
  sleipnir::Variable a = 10;
  sleipnir::Variable c = +a;

  EXPECT_DOUBLE_EQ(a.Value(), c.Value());
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
}

TEST(GradientTest, NegativeOperator) {
  sleipnir::Variable a = 10;
  sleipnir::Variable c = -a;

  EXPECT_DOUBLE_EQ(-a.Value(), c.Value());
  EXPECT_DOUBLE_EQ(-1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
}

TEST(GradientTest, IdenticalVariables) {
  sleipnir::Variable a = 10;
  sleipnir::Variable x = a;
  sleipnir::Variable c = a * a + x;

  EXPECT_DOUBLE_EQ(a.Value() * a.Value() + x.Value(), c.Value());
  EXPECT_DOUBLE_EQ(
      2 * a.Value() + sleipnir::Gradient(x, a).Calculate().coeff(0),
      sleipnir::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(
      2 * a.Value() * sleipnir::Gradient(x, a).Calculate().coeff(0) + 1,
      sleipnir::Gradient(c, a).Calculate().coeff(0));
}

TEST(GradientTest, Elementary) {
  sleipnir::Variable a = 1.0;
  sleipnir::Variable b = 2.0;
  sleipnir::Variable c = 3.0;

  c = -2 * a;
  EXPECT_DOUBLE_EQ(-2, sleipnir::Gradient(c, a).Calculate().coeff(0));

  c = a / 3.0;
  EXPECT_DOUBLE_EQ(1.0 / 3.0, sleipnir::Gradient(c, a).Calculate().coeff(0));

  a = 100.0;
  b = 200.0;

  c = a + b;
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, b).Calculate().coeff(0));

  c = a - b;
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(-1.0, sleipnir::Gradient(c, b).Calculate().coeff(0));

  c = -a + b;
  EXPECT_DOUBLE_EQ(-1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, b).Calculate().coeff(0));

  c = a + 1;
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(c, a).Calculate().coeff(0));
}

TEST(GradientTest, Comparison) {
  sleipnir::Variable x = 10.0;
  sleipnir::Variable a = 10.0;
  sleipnir::Variable b = 200.0;

  EXPECT_EQ(a, a);
  EXPECT_EQ(a, x);
  EXPECT_EQ(a, 10);
  EXPECT_EQ(10, a);

  EXPECT_NE(a, b);
  EXPECT_NE(a, 20);
  EXPECT_NE(20, a);

  EXPECT_LT(a, b);
  EXPECT_LT(a, 20);

  EXPECT_GT(b, a);
  EXPECT_GT(20, a);

  EXPECT_LE(a, a);
  EXPECT_LE(a, x);
  EXPECT_LE(a, b);
  EXPECT_LE(a, 10);
  EXPECT_LE(a, 20);

  EXPECT_GE(a, a);
  EXPECT_GE(x, a);
  EXPECT_GE(b, a);
  EXPECT_GE(10, a);
  EXPECT_GE(20, a);

  //------------------------------------------------------------------------------
  // TEST COMPARISON OPERATORS BETWEEN VARIABLE AND EXPRPTR
  //------------------------------------------------------------------------------
  EXPECT_EQ(a, a / a * a);
  EXPECT_EQ(a / a * a, a);

  EXPECT_NE(a, a - a);
  EXPECT_NE(a - a, a);

  EXPECT_LT(a - a, a);
  EXPECT_LT(a, a + a);

  EXPECT_GT(a + a, a);
  EXPECT_GT(a, a - a);

  EXPECT_LE(a, a - a + a);
  EXPECT_LE(a - a + a, a);

  EXPECT_LE(a, a + a);
  EXPECT_LE(a - a, a);

  EXPECT_GE(a, a - a + a);
  EXPECT_GE(a - a + a, a);

  EXPECT_GE(a + a, a);
  EXPECT_GE(a, a - a);
}

TEST(GradientTest, Trigonometry) {
  sleipnir::Variable x = 0.5;
  EXPECT_DOUBLE_EQ(std::sin(x.Value()), sleipnir::sin(x).Value());
  EXPECT_DOUBLE_EQ(
      std::cos(x.Value()),
      sleipnir::Gradient(sleipnir::sin(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::cos(x.Value()), sleipnir::cos(x).Value());
  EXPECT_DOUBLE_EQ(
      -std::sin(x.Value()),
      sleipnir::Gradient(sleipnir::cos(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::tan(x.Value()), sleipnir::tan(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / (std::cos(x.Value()) * std::cos(x.Value())),
      sleipnir::Gradient(sleipnir::tan(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::asin(x.Value()), sleipnir::asin(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / std::sqrt(1 - x.Value() * x.Value()),
      sleipnir::Gradient(sleipnir::asin(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::acos(x.Value()), sleipnir::acos(x).Value());
  EXPECT_DOUBLE_EQ(
      -1.0 / std::sqrt(1 - x.Value() * x.Value()),
      sleipnir::Gradient(sleipnir::acos(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::atan(x.Value()), sleipnir::atan(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / (1 + x.Value() * x.Value()),
      sleipnir::Gradient(sleipnir::atan(x), x).Calculate().coeff(0));
}

TEST(GradientTest, Hyperbolic) {
  sleipnir::Variable x = 1.0;
  EXPECT_DOUBLE_EQ(std::sinh(x.Value()), sleipnir::sinh(x).Value());
  EXPECT_DOUBLE_EQ(
      std::cosh(x.Value()),
      sleipnir::Gradient(sleipnir::sinh(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::cosh(x.Value()), sleipnir::cosh(x).Value());
  EXPECT_DOUBLE_EQ(
      std::sinh(x.Value()),
      sleipnir::Gradient(sleipnir::cosh(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::tanh(x.Value()), sleipnir::tanh(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / (std::cosh(x.Value()) * std::cosh(x.Value())),
      sleipnir::Gradient(sleipnir::tanh(x), x).Calculate().coeff(0));
}

TEST(GradientTest, Exponential) {
  sleipnir::Variable x = 1.0;
  EXPECT_DOUBLE_EQ(std::log(x.Value()), sleipnir::log(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / x.Value(),
      sleipnir::Gradient(sleipnir::log(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::log10(x.Value()), sleipnir::log10(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0 / (std::log(10) * x.Value()),
      sleipnir::Gradient(sleipnir::log10(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::exp(x.Value()), sleipnir::exp(x).Value());
  EXPECT_DOUBLE_EQ(
      std::exp(x.Value()),
      sleipnir::Gradient(sleipnir::exp(x), x).Calculate().coeff(0));
}

TEST(GradientTest, Power) {
  sleipnir::Variable x = 1.0;
  sleipnir::Variable a = 2.0;
  sleipnir::Variable y = 2 * a;
  EXPECT_DOUBLE_EQ(std::sqrt(x.Value()), sleipnir::sqrt(x).Value());
  EXPECT_DOUBLE_EQ(
      0.5 / std::sqrt(x.Value()),
      sleipnir::Gradient(sleipnir::sqrt(x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), 2.0), sleipnir::pow(x, 2.0).Value());
  EXPECT_DOUBLE_EQ(
      2.0 * x.Value(),
      sleipnir::Gradient(sleipnir::pow(x, 2.0), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(2.0, x.Value()), sleipnir::pow(2.0, x).Value());
  EXPECT_DOUBLE_EQ(
      std::log(2.0) * std::pow(2.0, x.Value()),
      sleipnir::Gradient(sleipnir::pow(2.0, x), x).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), x.Value()), sleipnir::pow(x, x).Value());
  EXPECT_DOUBLE_EQ(
      ((sleipnir::log(x) + 1) * sleipnir::pow(x, x)).Value(),
      sleipnir::Gradient(sleipnir::pow(x, x), x).Calculate().coeff(0));

  EXPECT_EQ(2 * a.Value(), y);
  EXPECT_DOUBLE_EQ(2.0, sleipnir::Gradient(y, a).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), y.Value()), sleipnir::pow(x, y).Value());
  EXPECT_DOUBLE_EQ(
      y.Value() / x.Value() * std::pow(x.Value(), y.Value()),
      sleipnir::Gradient(sleipnir::pow(x, y), x).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(
      std::pow(x.Value(), y.Value()) *
          (y.Value() / x.Value() *
               sleipnir::Gradient(x, a).Calculate().coeff(0) +
           std::log(x.Value()) * sleipnir::Gradient(y, a).Calculate().coeff(0)),
      sleipnir::Gradient(sleipnir::pow(x, y), a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(
      std::log(x.Value()) * std::pow(x.Value(), y.Value()),
      sleipnir::Gradient(sleipnir::pow(x, y), y).Calculate().coeff(0));
}

TEST(GradientTest, Abs) {
  sleipnir::Variable x = 1.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::abs(x).Value());
  EXPECT_DOUBLE_EQ(
      1.0, sleipnir::Gradient(sleipnir::abs(x), x).Calculate().coeff(0));
  x = -1.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::abs(x).Value());
  EXPECT_DOUBLE_EQ(
      -1.0, sleipnir::Gradient(sleipnir::abs(x), x).Calculate().coeff(0));
  x = 0.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::abs(x).Value());
  EXPECT_DOUBLE_EQ(
      0.0, sleipnir::Gradient(sleipnir::abs(x), x).Calculate().coeff(0));
}

TEST(GradientTest, Atan2) {
  // Testing atan2 function on (double, var)
  sleipnir::Variable x = 1.0;
  sleipnir::Variable y = 0.9;
  EXPECT_EQ(sleipnir::atan2(2.0, x).Value(), std::atan2(2.0, x.Value()));
  EXPECT_EQ(sleipnir::Gradient(sleipnir::atan2(2.0, x), x).Calculate().coeff(0),
            (-2.0 / (2 * 2 + x * x)).Value());

  // Testing atan2 function on (var, double)
  x = 1.0;
  EXPECT_EQ(sleipnir::atan2(x, 2.0), std::atan2(x.Value(), 2.0));
  EXPECT_EQ(sleipnir::Gradient(sleipnir::atan2(x, 2.0), x).Calculate().coeff(0),
            (2.0 / (2 * 2 + x * x)).Value());

  // Testing atan2 function on (var, var)
  x = 1.1;
  EXPECT_EQ(sleipnir::atan2(y, x), std::atan2(y.Value(), x.Value()));
  EXPECT_EQ(sleipnir::Gradient(sleipnir::atan2(y, x), y).Calculate().coeff(0),
            x / (x * x + y * y));
  EXPECT_EQ(sleipnir::Gradient(sleipnir::atan2(y, x), x).Calculate().coeff(0),
            -y / (x * x + y * y));

  // Testing atan2 function on (expr, expr)
  EXPECT_EQ(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1),
            3 * std::atan2(sleipnir::sin(y).Value(), 2 * x.Value() + 1));
  EXPECT_EQ(
      sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), y)
          .Calculate()
          .coeff(0),
      3 * (2 * x + 1) * sleipnir::cos(y) /
          ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)));
  EXPECT_EQ(
      sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), x)
          .Calculate()
          .coeff(0),
      3 * -2 * sleipnir::sin(y) /
          ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)));
}

TEST(GradientTest, Hypot) {
  // Testing hypot function on (var, double)
  sleipnir::Variable x = 1.8;
  sleipnir::Variable y = 1.5;
  EXPECT_EQ(std::hypot(x.Value(), 2.0), sleipnir::hypot(x, 2.0));
  EXPECT_EQ(
      x / std::hypot(x.Value(), 2.0),
      sleipnir::Gradient(sleipnir::hypot(x, 2.0), x).Calculate().coeff(0));

  // Testing hypot function on (double, var)
  EXPECT_EQ(std::hypot(2.0, y.Value()), sleipnir::hypot(2.0, y));
  EXPECT_EQ(
      y / std::hypot(2.0, y.Value()),
      sleipnir::Gradient(sleipnir::hypot(2.0, y), y).Calculate().coeff(0));

  // Testing hypot function on (var, var)
  x = 1.3;
  y = 2.3;
  EXPECT_EQ(std::hypot(x.Value(), y.Value()), sleipnir::hypot(x, y));
  EXPECT_EQ(x / std::hypot(x.Value(), y.Value()),
            sleipnir::Gradient(sleipnir::hypot(x, y), x).Calculate().coeff(0));
  EXPECT_EQ(y / std::hypot(x.Value(), y.Value()),
            sleipnir::Gradient(sleipnir::hypot(x, y), y).Calculate().coeff(0));

  // Testing hypot function on (expr, expr)
  x = 1.3;
  y = 2.3;
  EXPECT_EQ(std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::hypot(2.0 * x, 3.0 * y).Value());
  EXPECT_EQ(4.0 * x / std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), x)
                .Calculate()
                .coeff(0));
  EXPECT_EQ(9.0 * y / std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), y)
                .Calculate()
                .coeff(0));
}

TEST(GradientTest, Miscellaneous) {
  sleipnir::Variable x = 3.0;
  sleipnir::Variable y = x;

  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::abs(x).Value());
  EXPECT_DOUBLE_EQ(1.0, sleipnir::Gradient(x, x).Calculate().coeff(0));

  x = 0.5;
  EXPECT_DOUBLE_EQ(std::erf(x.Value()), sleipnir::erf(x).Value());
  EXPECT_EQ(2 / std::sqrt(std::numbers::pi) * std::exp(-x.Value() * x.Value()),
            sleipnir::Gradient(sleipnir::erf(x), x).Calculate().coeff(0));
}

TEST(GradientTest, Reuse) {
  sleipnir::Variable a = 10;
  sleipnir::Variable b = 20;
  sleipnir::Variable x = a * b;

  sleipnir::Gradient gradient{x, a};

  Eigen::VectorXd g = gradient.Calculate();
  EXPECT_EQ(20.0, g(0));

  b = 10;
  g = gradient.Calculate();
  EXPECT_EQ(10.0, g(0));
}
