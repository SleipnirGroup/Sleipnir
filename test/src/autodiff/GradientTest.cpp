// Copyright (c) Joshua Nichols and Tyler Veness

#include <numbers>

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Gradient.hpp>

TEST(GradientTest, Gradient) {
  sleipnir::autodiff::Variable a = 10;
  sleipnir::autodiff::Variable b = 20;
  sleipnir::autodiff::Variable c = a;
  sleipnir::autodiff::Variable x;
  sleipnir::autodiff::Variable y;
  sleipnir::autodiff::Variable r;

  //------------------------------------------------------------------------------
  // TEST TRIVIAL DERIVATIVE CALCULATIONS
  //------------------------------------------------------------------------------
  EXPECT_DOUBLE_EQ(1, sleipnir::autodiff::Gradient(a, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(0, sleipnir::autodiff::Gradient(a, b).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1, sleipnir::autodiff::Gradient(c, c).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(0, sleipnir::autodiff::Gradient(c, b).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST POSITIVE OPERATOR
  //------------------------------------------------------------------------------
  c = +a;

  EXPECT_DOUBLE_EQ(a.Value(), c.Value());
  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST NEGATIVE OPERATOR
  //------------------------------------------------------------------------------
  c = -a;

  EXPECT_DOUBLE_EQ(-a.Value(), c.Value());
  EXPECT_DOUBLE_EQ(-1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST WHEN IDENTICAL/EQUIVALENT VARIABLES ARE PRESENT
  //------------------------------------------------------------------------------
  x = a;
  c = a * a + x;

  EXPECT_DOUBLE_EQ(a.Value() * a.Value() + x.Value(), c.Value());
  EXPECT_DOUBLE_EQ(
      2 * a.Value() + sleipnir::autodiff::Gradient(x, a).Calculate().coeff(0),
      sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(
      2 * a.Value() * sleipnir::autodiff::Gradient(a, x).Calculate().coeff(0) +
          1,
      sleipnir::autodiff::Gradient(c, x).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST DERIVATIVES COMPUTATION AFTER CHANGING VAR VALUE
  //------------------------------------------------------------------------------
  a = sleipnir::autodiff::Variable{
      20.0};  // a is now a new independent variable

  EXPECT_DOUBLE_EQ(0.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(2 * x.Value() + 1,
                   sleipnir::autodiff::Gradient(c, x).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST MULTIPLICATION OPERATOR (USING CONSTANT FACTOR)
  //------------------------------------------------------------------------------
  c = -2 * a;

  EXPECT_DOUBLE_EQ(-2, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST DIVISION OPERATOR (USING CONSTANT FACTOR)
  //------------------------------------------------------------------------------
  c = a / 3.0;

  EXPECT_DOUBLE_EQ(1.0 / 3.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST DERIVATIVES WITH RESPECT TO DEPENDENT VARIABLES USING += -= *= /=
  //------------------------------------------------------------------------------

  a += 2.0;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a -= 3.0;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a *= 2.0;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a /= 3.0;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a += 2 * b;
  c = a * b;

  EXPECT_EQ(b + a * sleipnir::autodiff::Gradient(b, a).Calculate().coeff(0),
            sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a -= 3 * b;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a *= b;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  a /= b;
  c = a * b;

  EXPECT_EQ(b, sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST BINARY ARITHMETIC OPERATORS
  //------------------------------------------------------------------------------
  a = sleipnir::autodiff::Variable{100.0};
  b = sleipnir::autodiff::Variable{200.0};

  c = a + b;

  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, b).Calculate().coeff(0));

  c = a - b;

  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(-1.0,
                   sleipnir::autodiff::Gradient(c, b).Calculate().coeff(0));

  c = -a + b;

  EXPECT_DOUBLE_EQ(-1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));
  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, b).Calculate().coeff(0));

  c = a + 1;

  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(c, a).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST DERIVATIVES WITH RESPECT TO SUB-EXPRESSIONS
  //------------------------------------------------------------------------------
  x = 2 * a + b;
  r = x * x - a + b;

  EXPECT_EQ((2 * x).Value(),
            sleipnir::autodiff::Gradient(r, x).Calculate().coeff(0));
  EXPECT_EQ(
      (2 * x * sleipnir::autodiff::Gradient(x, a).Calculate().coeff(0) - 1.0)
          .Value(),
      sleipnir::autodiff::Gradient(r, a).Calculate().coeff(0));
  EXPECT_EQ(
      (2 * x * sleipnir::autodiff::Gradient(x, b).Calculate().coeff(0) + 1.0)
          .Value(),
      sleipnir::autodiff::Gradient(r, b).Calculate().coeff(0));

  //------------------------------------------------------------------------------
  // TEST COMPARISON OPERATORS
  //------------------------------------------------------------------------------
  a = 10;
  x = 10;

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

  //--------------------------------------------------------------------------
  // TEST TRIGONOMETRIC FUNCTIONS
  //--------------------------------------------------------------------------
  x = 0.5;

  EXPECT_DOUBLE_EQ(std::sin(x.Value()), sleipnir::autodiff::sin(x).Value());
  EXPECT_DOUBLE_EQ(std::cos(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::sin(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::cos(x.Value()), sleipnir::autodiff::cos(x).Value());
  EXPECT_DOUBLE_EQ(-std::sin(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::cos(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::tan(x.Value()), sleipnir::autodiff::tan(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / (std::cos(x.Value()) * std::cos(x.Value())),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::tan(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::asin(x.Value()), sleipnir::autodiff::asin(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / std::sqrt(1 - x.Value() * x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::asin(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::acos(x.Value()), sleipnir::autodiff::acos(x).Value());
  EXPECT_DOUBLE_EQ(-1.0 / std::sqrt(1 - x.Value() * x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::acos(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::atan(x.Value()), sleipnir::autodiff::atan(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / (1 + x.Value() * x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::atan(x), x)
                       .Calculate()
                       .coeff(0));

  //--------------------------------------------------------------------------
  // TEST HYPERBOLIC FUNCTIONS
  //--------------------------------------------------------------------------
  EXPECT_DOUBLE_EQ(std::sinh(x.Value()), sleipnir::autodiff::sinh(x).Value());
  EXPECT_DOUBLE_EQ(std::cosh(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::sinh(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::cosh(x.Value()), sleipnir::autodiff::cosh(x).Value());
  EXPECT_DOUBLE_EQ(std::sinh(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::cosh(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::tanh(x.Value()), sleipnir::autodiff::tanh(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / (std::cosh(x.Value()) * std::cosh(x.Value())),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::tanh(x), x)
                       .Calculate()
                       .coeff(0));

  //--------------------------------------------------------------------------
  // TEST EXPONENTIAL AND LOGARITHMIC FUNCTIONS
  //--------------------------------------------------------------------------
  EXPECT_DOUBLE_EQ(std::log(x.Value()), sleipnir::autodiff::log(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / x.Value(),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::log(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::log10(x.Value()), sleipnir::autodiff::log10(x).Value());
  EXPECT_DOUBLE_EQ(1.0 / (std::log(10) * x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::log10(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::exp(x.Value()), sleipnir::autodiff::exp(x).Value());
  EXPECT_DOUBLE_EQ(std::exp(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::exp(x), x)
                       .Calculate()
                       .coeff(0));

  //--------------------------------------------------------------------------
  // TEST POWER FUNCTIONS
  //--------------------------------------------------------------------------
  EXPECT_DOUBLE_EQ(std::sqrt(x.Value()), sleipnir::autodiff::sqrt(x).Value());
  EXPECT_DOUBLE_EQ(0.5 / std::sqrt(x.Value()),
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::sqrt(x), x)
                       .Calculate()
                       .coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), 2.0),
                   sleipnir::autodiff::pow(x, 2.0).Value());
  EXPECT_DOUBLE_EQ(2.0 * x.Value(), sleipnir::autodiff::Gradient(
                                        sleipnir::autodiff::pow(x, 2.0), x)
                                        .Calculate()
                                        .coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(2.0, x.Value()),
                   sleipnir::autodiff::pow(2.0, x).Value());
  EXPECT_DOUBLE_EQ(
      std::log(2.0) * std::pow(2.0, x.Value()),
      sleipnir::autodiff::Gradient(sleipnir::autodiff::pow(2.0, x), x)
          .Calculate()
          .coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), x.Value()),
                   sleipnir::autodiff::pow(x, x).Value());
  EXPECT_DOUBLE_EQ(
      ((sleipnir::autodiff::log(x) + 1) * sleipnir::autodiff::pow(x, x))
          .Value(),
      sleipnir::autodiff::Gradient(sleipnir::autodiff::pow(x, x), x)
          .Calculate()
          .coeff(0));

  y = 2 * a;

  EXPECT_EQ(2 * a.Value(), y);
  EXPECT_DOUBLE_EQ(2.0,
                   sleipnir::autodiff::Gradient(y, a).Calculate().coeff(0));

  EXPECT_DOUBLE_EQ(std::pow(x.Value(), y.Value()),
                   sleipnir::autodiff::pow(x, y).Value());
  EXPECT_DOUBLE_EQ(
      y.Value() / x.Value() * std::pow(x.Value(), y.Value()),
      sleipnir::autodiff::Gradient(sleipnir::autodiff::pow(x, y), x)
          .Calculate()
          .coeff(0));
  EXPECT_DOUBLE_EQ(
      std::pow(x.Value(), y.Value()) *
          (y.Value() / x.Value() *
               sleipnir::autodiff::Gradient(x, a).Calculate().coeff(0) +
           std::log(x.Value()) *
               sleipnir::autodiff::Gradient(y, a).Calculate().coeff(0)),
      sleipnir::autodiff::Gradient(sleipnir::autodiff::pow(x, y), a)
          .Calculate()
          .coeff(0));
  EXPECT_DOUBLE_EQ(
      std::log(x.Value()) * std::pow(x.Value(), y.Value()),
      sleipnir::autodiff::Gradient(sleipnir::autodiff::pow(x, y), y)
          .Calculate()
          .coeff(0));

  //--------------------------------------------------------------------------
  // TEST ABS FUNCTION
  //--------------------------------------------------------------------------

  x = 1.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::autodiff::abs(x).Value());
  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::abs(x), x)
                       .Calculate()
                       .coeff(0));
  x = -1.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::autodiff::abs(x).Value());
  EXPECT_DOUBLE_EQ(-1.0,
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::abs(x), x)
                       .Calculate()
                       .coeff(0));
  x = 0.0;
  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::autodiff::abs(x).Value());
  EXPECT_DOUBLE_EQ(0.0,
                   sleipnir::autodiff::Gradient(sleipnir::autodiff::abs(x), x)
                       .Calculate()
                       .coeff(0));

  //--------------------------------------------------------------------------
  // TEST ATAN2 FUNCTION
  //--------------------------------------------------------------------------

  // Testing atan2 function on (double, var)
  x = 1.0;
  EXPECT_EQ(sleipnir::autodiff::atan2(2.0, x).Value(),
            std::atan2(2.0, x.Value()));
  EXPECT_EQ(sleipnir::autodiff::Gradient(sleipnir::autodiff::atan2(2.0, x), x)
                .Calculate()
                .coeff(0),
            (-2.0 / (2 * 2 + x * x)).Value());

  // Testing atan2 function on (var, double)
  x = 1.0;
  EXPECT_EQ(sleipnir::autodiff::atan2(x, 2.0), std::atan2(x.Value(), 2.0));
  EXPECT_EQ(sleipnir::autodiff::Gradient(sleipnir::autodiff::atan2(x, 2.0), x)
                .Calculate()
                .coeff(0),
            (2.0 / (2 * 2 + x * x)).Value());

  // Testing atan2 function on (var, var)
  x = 1.1;
  y = 0.9;
  EXPECT_EQ(sleipnir::autodiff::atan2(y, x), std::atan2(y.Value(), x.Value()));
  EXPECT_EQ(sleipnir::autodiff::Gradient(sleipnir::autodiff::atan2(y, x), y)
                .Calculate()
                .coeff(0),
            x / (x * x + y * y));
  EXPECT_EQ(sleipnir::autodiff::Gradient(sleipnir::autodiff::atan2(y, x), x)
                .Calculate()
                .coeff(0),
            -y / (x * x + y * y));

  // Testing atan2 function on (expr, expr)
  EXPECT_EQ(
      3 * sleipnir::autodiff::atan2(sleipnir::autodiff::sin(y), 2 * x + 1),
      3 * std::atan2(sleipnir::autodiff::sin(y).Value(), 2 * x.Value() + 1));
  EXPECT_EQ(
      sleipnir::autodiff::Gradient(
          3 * sleipnir::autodiff::atan2(sleipnir::autodiff::sin(y), 2 * x + 1),
          y)
          .Calculate()
          .coeff(0),
      3 * (2 * x + 1) * sleipnir::autodiff::cos(y) /
          ((2 * x + 1) * (2 * x + 1) +
           sleipnir::autodiff::sin(y) * sleipnir::autodiff::sin(y)));
  EXPECT_EQ(
      sleipnir::autodiff::Gradient(
          3 * sleipnir::autodiff::atan2(sleipnir::autodiff::sin(y), 2 * x + 1),
          x)
          .Calculate()
          .coeff(0),
      3 * -2 * sleipnir::autodiff::sin(y) /
          ((2 * x + 1) * (2 * x + 1) +
           sleipnir::autodiff::sin(y) * sleipnir::autodiff::sin(y)));

  //--------------------------------------------------------------------------
  // TEST HYPOT2 FUNCTIONS
  //--------------------------------------------------------------------------

  // Testing hypot function on (var, double)
  x = 1.8;
  EXPECT_EQ(std::hypot(x.Value(), 2.0), sleipnir::autodiff::hypot(x, 2.0));
  EXPECT_EQ(x / std::hypot(x.Value(), 2.0),
            sleipnir::autodiff::Gradient(sleipnir::autodiff::hypot(x, 2.0), x)
                .Calculate()
                .coeff(0));

  // Testing hypot function on (double, var)
  y = 1.5;
  EXPECT_EQ(std::hypot(2.0, y.Value()), sleipnir::autodiff::hypot(2.0, y));
  EXPECT_EQ(y / std::hypot(2.0, y.Value()),
            sleipnir::autodiff::Gradient(sleipnir::autodiff::hypot(2.0, y), y)
                .Calculate()
                .coeff(0));

  // Testing hypot function on (var, var)
  x = 1.3;
  y = 2.3;
  EXPECT_EQ(std::hypot(x.Value(), y.Value()), sleipnir::autodiff::hypot(x, y));
  EXPECT_EQ(x / std::hypot(x.Value(), y.Value()),
            sleipnir::autodiff::Gradient(sleipnir::autodiff::hypot(x, y), x)
                .Calculate()
                .coeff(0));
  EXPECT_EQ(y / std::hypot(x.Value(), y.Value()),
            sleipnir::autodiff::Gradient(sleipnir::autodiff::hypot(x, y), y)
                .Calculate()
                .coeff(0));

  // Testing hypot function on (expr, expr)
  x = 1.3;
  y = 2.3;
  EXPECT_EQ(std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::autodiff::hypot(2.0 * x, 3.0 * y).Value());
  EXPECT_EQ(4.0 * x / std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::autodiff::Gradient(
                sleipnir::autodiff::hypot(2.0 * x, 3.0 * y), x)
                .Calculate()
                .coeff(0));
  EXPECT_EQ(9.0 * y / std::hypot(2.0 * x.Value(), 3.0 * y.Value()),
            sleipnir::autodiff::Gradient(
                sleipnir::autodiff::hypot(2.0 * x, 3.0 * y), y)
                .Calculate()
                .coeff(0));

  //--------------------------------------------------------------------------
  // TEST OTHER FUNCTIONS
  //--------------------------------------------------------------------------
  x = 3.0;
  y = x;

  EXPECT_DOUBLE_EQ(std::abs(x.Value()), sleipnir::autodiff::abs(x).Value());
  EXPECT_DOUBLE_EQ(1.0,
                   sleipnir::autodiff::Gradient(x, x).Calculate().coeff(0));

  x = 0.5;
  EXPECT_DOUBLE_EQ(std::erf(x.Value()), sleipnir::autodiff::erf(x).Value());
  EXPECT_EQ(2 / sleipnir::autodiff::sqrt(std::numbers::pi) *
                std::exp(-x.Value() * x.Value()),
            sleipnir::autodiff::Gradient(sleipnir::autodiff::erf(x), x)
                .Calculate()
                .coeff(0));
}

TEST(GradientTest, Reuse) {
  sleipnir::autodiff::Variable a = 10;
  sleipnir::autodiff::Variable b = 20;
  sleipnir::autodiff::Variable x = a * b;

  sleipnir::autodiff::Gradient gradient{x, a};

  auto g = gradient.Calculate();
  EXPECT_EQ(20.0, g.coeff(0));

  b = 10;
  g = gradient.Calculate();
  EXPECT_EQ(10.0, g.coeff(0));
}
