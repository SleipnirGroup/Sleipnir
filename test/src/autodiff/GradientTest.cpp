// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Gradient.hpp>

TEST_CASE("Gradient - Trivial case", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(10);
  sleipnir::Variable b;
  b.SetValue(20);
  sleipnir::Variable c = a;

  CHECK(sleipnir::Gradient(a, a).Value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(a, b).Value().coeff(0) == 0.0);
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).Value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - Unary plus", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(10);
  sleipnir::Variable c = +a;

  CHECK(c.Value() == a.Value());
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Unary minus", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(10);
  sleipnir::Variable c = -a;

  CHECK(c.Value() == -a.Value());
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == -1.0);
}

TEST_CASE("Gradient - Identical variables", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(10);
  sleipnir::Variable x = a;
  sleipnir::Variable c = a * a + x;

  CHECK(c.Value() == a.Value() * a.Value() + x.Value());
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) ==
        2 * a.Value() + sleipnir::Gradient(x, a).Value().coeff(0));
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) ==
        2 * a.Value() * sleipnir::Gradient(x, a).Value().coeff(0) + 1);
}

TEST_CASE("Gradient - Elementary", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(1.0);
  sleipnir::Variable b;
  b.SetValue(2.0);
  sleipnir::Variable c;
  c.SetValue(3.0);

  c = -2 * a;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == -2.0);

  c = a / 3.0;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0 / 3.0);

  a.SetValue(100.0);
  b.SetValue(200.0);

  c = a + b;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).Value().coeff(0) == 1.0);

  c = a - b;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).Value().coeff(0) == -1.0);

  c = -a + b;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == -1.0);
  CHECK(sleipnir::Gradient(c, b).Value().coeff(0) == 1.0);

  c = a + 1;
  CHECK(sleipnir::Gradient(c, a).Value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Comparison", "[Gradient]") {
  sleipnir::Variable x;
  x.SetValue(10.0);
  sleipnir::Variable a;
  a.SetValue(10.0);
  sleipnir::Variable b;
  b.SetValue(200.0);

  CHECK(a.Value() == a.Value());
  CHECK(a.Value() == x.Value());
  CHECK(a.Value() == 10.0);
  CHECK(10.0 == a.Value());

  CHECK(a.Value() != b.Value());
  CHECK(a.Value() != 20.0);
  CHECK(20.0 != a.Value());

  CHECK(a.Value() < b.Value());
  CHECK(a.Value() < 20.0);

  CHECK(b.Value() > a.Value());
  CHECK(20.0 > a.Value());

  CHECK(a.Value() <= a.Value());
  CHECK(a.Value() <= x.Value());
  CHECK(a.Value() <= b.Value());
  CHECK(a.Value() <= 10.0);
  CHECK(a.Value() <= 20.0);

  CHECK(a.Value() >= a.Value());
  CHECK(x.Value() >= a.Value());
  CHECK(b.Value() >= a.Value());
  CHECK(10.0 >= a.Value());
  CHECK(20.0 >= a.Value());

  // Comparison between variables and expressions
  CHECK(a.Value() == a.Value() / a.Value() * a.Value());
  CHECK(a.Value() / a.Value() * a.Value() == a.Value());

  CHECK(a.Value() != (a - a).Value());
  CHECK((a - a).Value() != a.Value());

  CHECK((a - a).Value() < a.Value());
  CHECK(a.Value() < (a + a).Value());

  CHECK((a + a).Value() > a.Value());
  CHECK(a.Value() > (a - a).Value());

  CHECK(a.Value() <= (a - a + a).Value());
  CHECK((a - a + a).Value() <= a.Value());

  CHECK(a.Value() <= (a + a).Value());
  CHECK((a - a).Value() <= a.Value());

  CHECK(a.Value() >= (a - a + a).Value());
  CHECK((a - a + a).Value() >= a.Value());

  CHECK((a + a).Value() >= a.Value());
  CHECK(a.Value() >= (a - a).Value());
}

TEST_CASE("Gradient - Trigonometry", "[Gradient]") {
  sleipnir::Variable x;
  x.SetValue(0.5);

  // std::sin(x)
  CHECK(sleipnir::sin(x).Value() == std::sin(x.Value()));

  auto g = sleipnir::Gradient(sleipnir::sin(x), x);
  CHECK(g.Get().Value().coeff(0) == std::cos(x.Value()));
  CHECK(g.Value().coeff(0) == std::cos(x.Value()));

  // std::cos(x)
  CHECK(sleipnir::cos(x).Value() == std::cos(x.Value()));

  g = sleipnir::Gradient(sleipnir::cos(x), x);
  CHECK(g.Get().Value().coeff(0) == -std::sin(x.Value()));
  CHECK(g.Value().coeff(0) == -std::sin(x.Value()));

  // std::tan(x)
  CHECK(sleipnir::tan(x).Value() == std::tan(x.Value()));

  g = sleipnir::Gradient(sleipnir::tan(x), x);
  CHECK(g.Get().Value().coeff(0) ==
        1.0 / (std::cos(x.Value()) * std::cos(x.Value())));
  CHECK(g.Value().coeff(0) ==
        1.0 / (std::cos(x.Value()) * std::cos(x.Value())));

  // std::asin(x)
  CHECK(sleipnir::asin(x).Value() == std::asin(x.Value()));

  g = sleipnir::Gradient(sleipnir::asin(x), x);
  CHECK(g.Get().Value().coeff(0) == 1.0 / std::sqrt(1 - x.Value() * x.Value()));
  CHECK(g.Value().coeff(0) == 1.0 / std::sqrt(1 - x.Value() * x.Value()));

  // std::acos(x)
  CHECK(sleipnir::acos(x).Value() == std::acos(x.Value()));

  g = sleipnir::Gradient(sleipnir::acos(x), x);
  CHECK(g.Get().Value().coeff(0) ==
        -1.0 / std::sqrt(1 - x.Value() * x.Value()));
  CHECK(g.Value().coeff(0) == -1.0 / std::sqrt(1 - x.Value() * x.Value()));

  // std::atan(x)
  CHECK(sleipnir::atan(x).Value() == std::atan(x.Value()));

  g = sleipnir::Gradient(sleipnir::atan(x), x);
  CHECK(g.Get().Value().coeff(0) == 1.0 / (1 + x.Value() * x.Value()));
  CHECK(g.Value().coeff(0) == 1.0 / (1 + x.Value() * x.Value()));
}

TEST_CASE("Gradient - Hyperbolic", "[Gradient]") {
  sleipnir::Variable x;
  x.SetValue(1.0);

  // sinh(x)
  CHECK(sleipnir::sinh(x).Value() == std::sinh(x.Value()));

  auto g = sleipnir::Gradient(sleipnir::sinh(x), x);
  CHECK(g.Get().Value().coeff(0) == std::cosh(x.Value()));
  CHECK(g.Value().coeff(0) == std::cosh(x.Value()));

  // std::cosh(x)
  CHECK(sleipnir::cosh(x).Value() == std::cosh(x.Value()));

  g = sleipnir::Gradient(sleipnir::cosh(x), x);
  CHECK(g.Get().Value().coeff(0) == std::sinh(x.Value()));
  CHECK(g.Value().coeff(0) == std::sinh(x.Value()));

  // tanh(x)
  CHECK(sleipnir::tanh(x).Value() == std::tanh(x.Value()));

  g = sleipnir::Gradient(sleipnir::tanh(x), x);
  CHECK(g.Get().Value().coeff(0) ==
        1.0 / (std::cosh(x.Value()) * std::cosh(x.Value())));
  CHECK(g.Value().coeff(0) ==
        1.0 / (std::cosh(x.Value()) * std::cosh(x.Value())));
}

TEST_CASE("Gradient - Exponential", "[Gradient]") {
  sleipnir::Variable x;
  x.SetValue(1.0);

  // std::log(x)
  CHECK(sleipnir::log(x).Value() == std::log(x.Value()));

  auto g = sleipnir::Gradient(sleipnir::log(x), x);
  CHECK(g.Get().Value().coeff(0) == 1.0 / x.Value());
  CHECK(g.Value().coeff(0) == 1.0 / x.Value());

  // std::log10(x)
  CHECK(sleipnir::log10(x).Value() == std::log10(x.Value()));

  g = sleipnir::Gradient(sleipnir::log10(x), x);
  CHECK(g.Get().Value().coeff(0) == 1.0 / (std::log(10.0) * x.Value()));
  CHECK(g.Value().coeff(0) == 1.0 / (std::log(10.0) * x.Value()));

  // std::exp(x)
  CHECK(sleipnir::exp(x).Value() == std::exp(x.Value()));

  g = sleipnir::Gradient(sleipnir::exp(x), x);
  CHECK(g.Get().Value().coeff(0) == std::exp(x.Value()));
  CHECK(g.Value().coeff(0) == std::exp(x.Value()));
}

TEST_CASE("Gradient - Power", "[Gradient]") {
  sleipnir::Variable x;
  x.SetValue(1.0);
  sleipnir::Variable a;
  a.SetValue(2.0);
  sleipnir::Variable y = 2 * a;

  // std::sqrt(x)
  CHECK(sleipnir::sqrt(x).Value() == std::sqrt(x.Value()));

  auto g = sleipnir::Gradient(sleipnir::sqrt(x), x);
  CHECK(g.Get().Value().coeff(0) == 0.5 / std::sqrt(x.Value()));
  CHECK(g.Value().coeff(0) == 0.5 / std::sqrt(x.Value()));

  // x²
  CHECK(sleipnir::pow(x, 2.0).Value() == std::pow(x.Value(), 2.0));

  g = sleipnir::Gradient(sleipnir::pow(x, 2.0), x);
  CHECK(g.Get().Value().coeff(0) == 2.0 * x.Value());
  CHECK(g.Value().coeff(0) == 2.0 * x.Value());

  // 2ˣ
  CHECK(sleipnir::pow(2.0, x).Value() == std::pow(2.0, x.Value()));

  g = sleipnir::Gradient(sleipnir::pow(2.0, x), x);
  CHECK(g.Get().Value().coeff(0) == std::log(2.0) * std::pow(2.0, x.Value()));
  CHECK(g.Value().coeff(0) == std::log(2.0) * std::pow(2.0, x.Value()));

  // xˣ
  CHECK(sleipnir::pow(x, x).Value() == std::pow(x.Value(), x.Value()));

  g = sleipnir::Gradient(sleipnir::pow(x, x), x);
  CHECK(g.Get().Value().coeff(0) ==
        ((sleipnir::log(x) + 1) * sleipnir::pow(x, x)).Value());
  CHECK(g.Value().coeff(0) ==
        ((sleipnir::log(x) + 1) * sleipnir::pow(x, x)).Value());

  // y(a)
  CHECK(y.Value() == 2 * a.Value());

  g = sleipnir::Gradient(y, a);
  CHECK(g.Get().Value().coeff(0) == 2.0);
  CHECK(g.Value().coeff(0) == 2.0);

  // xʸ(x)
  CHECK(sleipnir::pow(x, y).Value() == std::pow(x.Value(), y.Value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), x);
  CHECK(g.Get().Value().coeff(0) ==
        y.Value() / x.Value() * std::pow(x.Value(), y.Value()));
  CHECK(g.Value().coeff(0) ==
        y.Value() / x.Value() * std::pow(x.Value(), y.Value()));

  // xʸ(a)
  CHECK(sleipnir::pow(x, y).Value() == std::pow(x.Value(), y.Value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), a);
  CHECK(g.Get().Value().coeff(0) ==
        std::pow(x.Value(), y.Value()) *
            (y.Value() / x.Value() * sleipnir::Gradient(x, a).Value().coeff(0) +
             std::log(x.Value()) * sleipnir::Gradient(y, a).Value().coeff(0)));
  CHECK(g.Value().coeff(0) ==
        std::pow(x.Value(), y.Value()) *
            (y.Value() / x.Value() * sleipnir::Gradient(x, a).Value().coeff(0) +
             std::log(x.Value()) * sleipnir::Gradient(y, a).Value().coeff(0)));

  // xʸ(y)
  CHECK(sleipnir::pow(x, y).Value() == std::pow(x.Value(), y.Value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), y);
  CHECK(g.Get().Value().coeff(0) ==
        std::log(x.Value()) * std::pow(x.Value(), y.Value()));
  CHECK(g.Value().coeff(0) ==
        std::log(x.Value()) * std::pow(x.Value(), y.Value()));
}

TEST_CASE("Gradient - std::abs()", "[Gradient]") {
  sleipnir::Variable x;
  auto g = sleipnir::Gradient(sleipnir::abs(x), x);

  x.SetValue(1.0);
  CHECK(sleipnir::abs(x).Value() == std::abs(x.Value()));
  CHECK(g.Get().Value().coeff(0) == 1.0);
  CHECK(g.Value().coeff(0) == 1.0);

  x.SetValue(-1.0);
  CHECK(sleipnir::abs(x).Value() == std::abs(x.Value()));
  CHECK(g.Get().Value().coeff(0) == -1.0);
  CHECK(g.Value().coeff(0) == -1.0);

  x.SetValue(0.0);
  CHECK(sleipnir::abs(x).Value() == std::abs(x.Value()));
  CHECK(g.Get().Value().coeff(0) == 0.0);
  CHECK(g.Value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - std::atan2()", "[Gradient]") {
  sleipnir::Variable x;
  sleipnir::Variable y;

  // Testing atan2 function on (double, var)
  x.SetValue(1.0);
  y.SetValue(0.9);
  CHECK(sleipnir::atan2(2.0, x).Value() == std::atan2(2.0, x.Value()));

  auto g = sleipnir::Gradient(sleipnir::atan2(2.0, x), x);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx((-2.0 / (2 * 2 + x * x)).Value()).margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx((-2.0 / (2 * 2 + x * x)).Value()).margin(1e-15));

  // Testing atan2 function on (var, double)
  x.SetValue(1.0);
  y.SetValue(0.9);
  CHECK(sleipnir::atan2(x, 2.0).Value() == std::atan2(x.Value(), 2.0));

  g = sleipnir::Gradient(sleipnir::atan2(x, 2.0), x);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx((2.0 / (2 * 2 + x * x)).Value()).margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx((2.0 / (2 * 2 + x * x)).Value()).margin(1e-15));

  // Testing atan2 function on (var, var)
  x.SetValue(1.1);
  y.SetValue(0.9);
  CHECK(sleipnir::atan2(y, x).Value() == std::atan2(y.Value(), x.Value()));

  g = sleipnir::Gradient(sleipnir::atan2(y, x), y);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx((x / (x * x + y * y)).Value()).margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx((x / (x * x + y * y)).Value()).margin(1e-15));

  g = sleipnir::Gradient(sleipnir::atan2(y, x), x);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx((-y / (x * x + y * y)).Value()).margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx((-y / (x * x + y * y)).Value()).margin(1e-15));

  // Testing atan2 function on (expr, expr)
  CHECK(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1).Value() ==
        3 * std::atan2(sleipnir::sin(y).Value(), 2 * x.Value() + 1));

  g = sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), y);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx(
            (3 * (2 * x + 1) * sleipnir::cos(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .Value())
            .margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx(
            (3 * (2 * x + 1) * sleipnir::cos(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .Value())
            .margin(1e-15));

  g = sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), x);
  CHECK(g.Get().Value().coeff(0) ==
        Catch::Approx(
            (3 * -2 * sleipnir::sin(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .Value())
            .margin(1e-15));
  CHECK(g.Value().coeff(0) ==
        Catch::Approx(
            (3 * -2 * sleipnir::sin(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .Value())
            .margin(1e-15));
}

TEST_CASE("Gradient - std::hypot()", "[Gradient]") {
  sleipnir::Variable x;
  sleipnir::Variable y;

  // Testing hypot function on (var, double)
  x.SetValue(1.8);
  y.SetValue(1.5);
  CHECK(sleipnir::hypot(x, 2.0).Value() == std::hypot(x.Value(), 2.0));

  auto g = sleipnir::Gradient(sleipnir::hypot(x, 2.0), x);
  CHECK(g.Get().Value().coeff(0) == (x / std::hypot(x.Value(), 2.0)).Value());
  CHECK(g.Value().coeff(0) == (x / std::hypot(x.Value(), 2.0)).Value());

  // Testing hypot function on (double, var)
  CHECK(sleipnir::hypot(2.0, y).Value() == std::hypot(2.0, y.Value()));

  g = sleipnir::Gradient(sleipnir::hypot(2.0, y), y);
  CHECK(g.Get().Value().coeff(0) == (y / std::hypot(2.0, y.Value())).Value());
  CHECK(g.Value().coeff(0) == (y / std::hypot(2.0, y.Value())).Value());

  // Testing hypot function on (var, var)
  x.SetValue(1.3);
  y.SetValue(2.3);
  CHECK(sleipnir::hypot(x, y).Value() == std::hypot(x.Value(), y.Value()));

  g = sleipnir::Gradient(sleipnir::hypot(x, y), x);
  CHECK(g.Get().Value().coeff(0) ==
        (x / std::hypot(x.Value(), y.Value())).Value());
  CHECK(g.Value().coeff(0) == (x / std::hypot(x.Value(), y.Value())).Value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y), y);
  CHECK(g.Get().Value().coeff(0) ==
        (y / std::hypot(x.Value(), y.Value())).Value());
  CHECK(g.Value().coeff(0) == (y / std::hypot(x.Value(), y.Value())).Value());

  // Testing hypot function on (expr, expr)
  x.SetValue(1.3);
  y.SetValue(2.3);
  CHECK(sleipnir::hypot(2.0 * x, 3.0 * y).Value() ==
        std::hypot(2.0 * x.Value(), 3.0 * y.Value()));

  g = sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), x);
  CHECK(g.Get().Value().coeff(0) ==
        (4.0 * x / std::hypot(2.0 * x.Value(), 3.0 * y.Value())).Value());
  CHECK(g.Value().coeff(0) ==
        (4.0 * x / std::hypot(2.0 * x.Value(), 3.0 * y.Value())).Value());

  g = sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), y);
  CHECK(g.Get().Value().coeff(0) ==
        (9.0 * y / std::hypot(2.0 * x.Value(), 3.0 * y.Value())).Value());
  CHECK(g.Value().coeff(0) ==
        (9.0 * y / std::hypot(2.0 * x.Value(), 3.0 * y.Value())).Value());

  // Testing hypot function on (var, var, var)
  sleipnir::Variable z;
  x.SetValue(1.3);
  y.SetValue(2.3);
  z.SetValue(3.3);
  CHECK(sleipnir::hypot(x, y, z).Value() ==
        std::hypot(x.Value(), y.Value(), z.Value()));

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), x);
  CHECK(g.Get().Value().coeff(0) ==
        (x / std::hypot(x.Value(), y.Value(), z.Value())).Value());
  CHECK(g.Value().coeff(0) ==
        (x / std::hypot(x.Value(), y.Value(), z.Value())).Value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), y);
  CHECK(g.Get().Value().coeff(0) ==
        (y / std::hypot(x.Value(), y.Value(), z.Value())).Value());
  CHECK(g.Value().coeff(0) ==
        (y / std::hypot(x.Value(), y.Value(), z.Value())).Value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), z);
  CHECK(g.Get().Value().coeff(0) ==
        (z / std::hypot(x.Value(), y.Value(), z.Value())).Value());
  CHECK(g.Value().coeff(0) ==
        (z / std::hypot(x.Value(), y.Value(), z.Value())).Value());
}

TEST_CASE("Gradient - Miscellaneous", "[Gradient]") {
  sleipnir::Variable x;

  // dx/dx
  x.SetValue(3.0);
  CHECK(sleipnir::abs(x).Value() == std::abs(x.Value()));

  auto g = sleipnir::Gradient(x, x);
  CHECK(g.Get().Value().coeff(0) == 1.0);
  CHECK(g.Value().coeff(0) == 1.0);

  // std::erf(x)
  x.SetValue(0.5);
  CHECK(sleipnir::erf(x).Value() == std::erf(x.Value()));

  g = sleipnir::Gradient(sleipnir::erf(x), x);
  CHECK(g.Get().Value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.Value() * x.Value()));
  CHECK(g.Value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.Value() * x.Value()));
}

TEST_CASE("Gradient - Variable reuse", "[Gradient]") {
  sleipnir::Variable a;
  a.SetValue(10);

  sleipnir::Variable b;
  b.SetValue(20);

  sleipnir::Variable x = a * b;

  auto g = sleipnir::Gradient(x, a);

  CHECK(g.Get().Value().coeff(0) == 20.0);
  CHECK(g.Value().coeff(0) == 20.0);

  b.SetValue(10);
  CHECK(g.Get().Value().coeff(0) == 10.0);
  CHECK(g.Value().coeff(0) == 10.0);
}

TEST_CASE("Gradient - sign()", "[Gradient]") {
  auto sign = [](double x) {
    if (x < 0.0) {
      return -1.0;
    } else if (x == 0.0) {
      return 0.0;
    } else {
      return 1.0;
    }
  };

  sleipnir::Variable x;

  // sgn(1.0)
  x.SetValue(1.0);
  CHECK(sleipnir::sign(x).Value() == sign(x.Value()));

  auto g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.Get().Value().coeff(0) == 0.0);
  CHECK(g.Value().coeff(0) == 0.0);

  // sgn(-1.0)
  x.SetValue(-1.0);
  CHECK(sleipnir::sign(x).Value() == sign(x.Value()));

  g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.Get().Value().coeff(0) == 0.0);
  CHECK(g.Value().coeff(0) == 0.0);

  // sgn(0.0)
  x.SetValue(0.0);
  CHECK(sleipnir::sign(x).Value() == sign(x.Value()));

  g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.Get().Value().coeff(0) == 0.0);
  CHECK(g.Value().coeff(0) == 0.0);
}
