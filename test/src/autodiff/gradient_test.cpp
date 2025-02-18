// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>

TEST_CASE("Gradient - Trivial case", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(10);
  sleipnir::Variable b;
  b.set_value(20);
  sleipnir::Variable c = a;

  CHECK(sleipnir::Gradient(a, a).value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(a, b).value().coeff(0) == 0.0);
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - Unary plus", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(10);
  sleipnir::Variable c = +a;

  CHECK(c.value() == a.value());
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Unary minus", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(10);
  sleipnir::Variable c = -a;

  CHECK(c.value() == -a.value());
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == -1.0);
}

TEST_CASE("Gradient - Identical variables", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(10);
  sleipnir::Variable x = a;
  sleipnir::Variable c = a * a + x;

  CHECK(c.value() == a.value() * a.value() + x.value());
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) ==
        2 * a.value() + sleipnir::Gradient(x, a).value().coeff(0));
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) ==
        2 * a.value() * sleipnir::Gradient(x, a).value().coeff(0) + 1);
}

TEST_CASE("Gradient - Elementary", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(1.0);
  sleipnir::Variable b;
  b.set_value(2.0);
  sleipnir::Variable c;
  c.set_value(3.0);

  c = -2 * a;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == -2.0);

  c = a / 3.0;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0 / 3.0);

  a.set_value(100.0);
  b.set_value(200.0);

  c = a + b;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).value().coeff(0) == 1.0);

  c = a - b;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(sleipnir::Gradient(c, b).value().coeff(0) == -1.0);

  c = -a + b;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == -1.0);
  CHECK(sleipnir::Gradient(c, b).value().coeff(0) == 1.0);

  c = a + 1;
  CHECK(sleipnir::Gradient(c, a).value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Comparison", "[Gradient]") {
  sleipnir::Variable x;
  x.set_value(10.0);
  sleipnir::Variable a;
  a.set_value(10.0);
  sleipnir::Variable b;
  b.set_value(200.0);

  CHECK(a.value() == a.value());
  CHECK(a.value() == x.value());
  CHECK(a.value() == 10.0);
  CHECK(10.0 == a.value());

  CHECK(a.value() != b.value());
  CHECK(a.value() != 20.0);
  CHECK(20.0 != a.value());

  CHECK(a.value() < b.value());
  CHECK(a.value() < 20.0);

  CHECK(b.value() > a.value());
  CHECK(20.0 > a.value());

  CHECK(a.value() <= a.value());
  CHECK(a.value() <= x.value());
  CHECK(a.value() <= b.value());
  CHECK(a.value() <= 10.0);
  CHECK(a.value() <= 20.0);

  CHECK(a.value() >= a.value());
  CHECK(x.value() >= a.value());
  CHECK(b.value() >= a.value());
  CHECK(10.0 >= a.value());
  CHECK(20.0 >= a.value());

  // Comparison between variables and expressions
  CHECK(a.value() == a.value() / a.value() * a.value());
  CHECK(a.value() / a.value() * a.value() == a.value());

  CHECK(a.value() != (a - a).value());
  CHECK((a - a).value() != a.value());

  CHECK((a - a).value() < a.value());
  CHECK(a.value() < (a + a).value());

  CHECK((a + a).value() > a.value());
  CHECK(a.value() > (a - a).value());

  CHECK(a.value() <= (a - a + a).value());
  CHECK((a - a + a).value() <= a.value());

  CHECK(a.value() <= (a + a).value());
  CHECK((a - a).value() <= a.value());

  CHECK(a.value() >= (a - a + a).value());
  CHECK((a - a + a).value() >= a.value());

  CHECK((a + a).value() >= a.value());
  CHECK(a.value() >= (a - a).value());
}

TEST_CASE("Gradient - Trigonometry", "[Gradient]") {
  sleipnir::Variable x;
  x.set_value(0.5);

  // std::sin(x)
  CHECK(sleipnir::sin(x).value() == std::sin(x.value()));

  auto g = sleipnir::Gradient(sleipnir::sin(x), x);
  CHECK(g.get().value().coeff(0) == std::cos(x.value()));
  CHECK(g.value().coeff(0) == std::cos(x.value()));

  // std::cos(x)
  CHECK(sleipnir::cos(x).value() == std::cos(x.value()));

  g = sleipnir::Gradient(sleipnir::cos(x), x);
  CHECK(g.get().value().coeff(0) == -std::sin(x.value()));
  CHECK(g.value().coeff(0) == -std::sin(x.value()));

  // std::tan(x)
  CHECK(sleipnir::tan(x).value() == std::tan(x.value()));

  g = sleipnir::Gradient(sleipnir::tan(x), x);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (std::cos(x.value()) * std::cos(x.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (std::cos(x.value()) * std::cos(x.value())));

  // std::asin(x)
  CHECK(sleipnir::asin(x).value() == std::asin(x.value()));

  g = sleipnir::Gradient(sleipnir::asin(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / std::sqrt(1 - x.value() * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / std::sqrt(1 - x.value() * x.value()));

  // std::acos(x)
  CHECK(sleipnir::acos(x).value() == std::acos(x.value()));

  g = sleipnir::Gradient(sleipnir::acos(x), x);
  CHECK(g.get().value().coeff(0) ==
        -1.0 / std::sqrt(1 - x.value() * x.value()));
  CHECK(g.value().coeff(0) == -1.0 / std::sqrt(1 - x.value() * x.value()));

  // std::atan(x)
  CHECK(sleipnir::atan(x).value() == std::atan(x.value()));

  g = sleipnir::Gradient(sleipnir::atan(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / (1 + x.value() * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / (1 + x.value() * x.value()));
}

TEST_CASE("Gradient - Hyperbolic", "[Gradient]") {
  sleipnir::Variable x;
  x.set_value(1.0);

  // sinh(x)
  CHECK(sleipnir::sinh(x).value() == std::sinh(x.value()));

  auto g = sleipnir::Gradient(sleipnir::sinh(x), x);
  CHECK(g.get().value().coeff(0) == std::cosh(x.value()));
  CHECK(g.value().coeff(0) == std::cosh(x.value()));

  // std::cosh(x)
  CHECK(sleipnir::cosh(x).value() == std::cosh(x.value()));

  g = sleipnir::Gradient(sleipnir::cosh(x), x);
  CHECK(g.get().value().coeff(0) == std::sinh(x.value()));
  CHECK(g.value().coeff(0) == std::sinh(x.value()));

  // tanh(x)
  CHECK(sleipnir::tanh(x).value() == std::tanh(x.value()));

  g = sleipnir::Gradient(sleipnir::tanh(x), x);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (std::cosh(x.value()) * std::cosh(x.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (std::cosh(x.value()) * std::cosh(x.value())));
}

TEST_CASE("Gradient - Exponential", "[Gradient]") {
  sleipnir::Variable x;
  x.set_value(1.0);

  // std::log(x)
  CHECK(sleipnir::log(x).value() == std::log(x.value()));

  auto g = sleipnir::Gradient(sleipnir::log(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / x.value());
  CHECK(g.value().coeff(0) == 1.0 / x.value());

  // std::log10(x)
  CHECK(sleipnir::log10(x).value() == std::log10(x.value()));

  g = sleipnir::Gradient(sleipnir::log10(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / (std::log(10.0) * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / (std::log(10.0) * x.value()));

  // std::exp(x)
  CHECK(sleipnir::exp(x).value() == std::exp(x.value()));

  g = sleipnir::Gradient(sleipnir::exp(x), x);
  CHECK(g.get().value().coeff(0) == std::exp(x.value()));
  CHECK(g.value().coeff(0) == std::exp(x.value()));
}

TEST_CASE("Gradient - Power", "[Gradient]") {
  sleipnir::Variable x;
  x.set_value(1.0);
  sleipnir::Variable a;
  a.set_value(2.0);
  sleipnir::Variable y = 2 * a;

  // std::sqrt(x)
  CHECK(sleipnir::sqrt(x).value() == std::sqrt(x.value()));

  auto g = sleipnir::Gradient(sleipnir::sqrt(x), x);
  CHECK(g.get().value().coeff(0) == 0.5 / std::sqrt(x.value()));
  CHECK(g.value().coeff(0) == 0.5 / std::sqrt(x.value()));

  // std::sqrt(a)
  CHECK(sleipnir::sqrt(a).value() == std::sqrt(a.value()));

  g = sleipnir::Gradient(sleipnir::sqrt(a), a);
  CHECK(g.get().value().coeff(0) == 0.5 / std::sqrt(a.value()));
  CHECK(g.value().coeff(0) == 0.5 / std::sqrt(a.value()));

  // x²
  CHECK(sleipnir::pow(x, 2.0).value() == std::pow(x.value(), 2.0));

  g = sleipnir::Gradient(sleipnir::pow(x, 2.0), x);
  CHECK(g.get().value().coeff(0) == 2.0 * x.value());
  CHECK(g.value().coeff(0) == 2.0 * x.value());

  // 2ˣ
  CHECK(sleipnir::pow(2.0, x).value() == std::pow(2.0, x.value()));

  g = sleipnir::Gradient(sleipnir::pow(2.0, x), x);
  CHECK(g.get().value().coeff(0) == std::log(2.0) * std::pow(2.0, x.value()));
  CHECK(g.value().coeff(0) == std::log(2.0) * std::pow(2.0, x.value()));

  // xˣ
  CHECK(sleipnir::pow(x, x).value() == std::pow(x.value(), x.value()));

  g = sleipnir::Gradient(sleipnir::pow(x, x), x);
  CHECK(g.get().value().coeff(0) ==
        ((sleipnir::log(x) + 1) * sleipnir::pow(x, x)).value());
  CHECK(g.value().coeff(0) ==
        ((sleipnir::log(x) + 1) * sleipnir::pow(x, x)).value());

  // y(a)
  CHECK(y.value() == 2 * a.value());

  g = sleipnir::Gradient(y, a);
  CHECK(g.get().value().coeff(0) == 2.0);
  CHECK(g.value().coeff(0) == 2.0);

  // xʸ(x)
  CHECK(sleipnir::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        y.value() / x.value() * std::pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) ==
        y.value() / x.value() * std::pow(x.value(), y.value()));

  // xʸ(a)
  CHECK(sleipnir::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), a);
  CHECK(g.get().value().coeff(0) ==
        std::pow(x.value(), y.value()) *
            (y.value() / x.value() * sleipnir::Gradient(x, a).value().coeff(0) +
             std::log(x.value()) * sleipnir::Gradient(y, a).value().coeff(0)));
  CHECK(g.value().coeff(0) ==
        std::pow(x.value(), y.value()) *
            (y.value() / x.value() * sleipnir::Gradient(x, a).value().coeff(0) +
             std::log(x.value()) * sleipnir::Gradient(y, a).value().coeff(0)));

  // xʸ(y)
  CHECK(sleipnir::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = sleipnir::Gradient(sleipnir::pow(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        std::log(x.value()) * std::pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) ==
        std::log(x.value()) * std::pow(x.value(), y.value()));
}

TEST_CASE("Gradient - std::abs()", "[Gradient]") {
  sleipnir::Variable x;
  auto g = sleipnir::Gradient(sleipnir::abs(x), x);

  x.set_value(1.0);
  CHECK(sleipnir::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == 1.0);
  CHECK(g.value().coeff(0) == 1.0);

  x.set_value(-1.0);
  CHECK(sleipnir::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == -1.0);
  CHECK(g.value().coeff(0) == -1.0);

  x.set_value(0.0);
  CHECK(sleipnir::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - std::atan2()", "[Gradient]") {
  sleipnir::Variable x;
  sleipnir::Variable y;

  // Testing atan2 function on (double, var)
  x.set_value(1.0);
  y.set_value(0.9);
  CHECK(sleipnir::atan2(2.0, x).value() == std::atan2(2.0, x.value()));

  auto g = sleipnir::Gradient(sleipnir::atan2(2.0, x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx((-2.0 / (2 * 2 + x * x)).value()).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx((-2.0 / (2 * 2 + x * x)).value()).margin(1e-15));

  // Testing atan2 function on (var, double)
  x.set_value(1.0);
  y.set_value(0.9);
  CHECK(sleipnir::atan2(x, 2.0).value() == std::atan2(x.value(), 2.0));

  g = sleipnir::Gradient(sleipnir::atan2(x, 2.0), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx((2.0 / (2 * 2 + x * x)).value()).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx((2.0 / (2 * 2 + x * x)).value()).margin(1e-15));

  // Testing atan2 function on (var, var)
  x.set_value(1.1);
  y.set_value(0.9);
  CHECK(sleipnir::atan2(y, x).value() == std::atan2(y.value(), x.value()));

  g = sleipnir::Gradient(sleipnir::atan2(y, x), y);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx((x / (x * x + y * y)).value()).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx((x / (x * x + y * y)).value()).margin(1e-15));

  g = sleipnir::Gradient(sleipnir::atan2(y, x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx((-y / (x * x + y * y)).value()).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx((-y / (x * x + y * y)).value()).margin(1e-15));

  // Testing atan2 function on (expr, expr)
  CHECK(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1).value() ==
        3 * std::atan2(sleipnir::sin(y).value(), 2 * x.value() + 1));

  g = sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), y);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(
            (3 * (2 * x + 1) * sleipnir::cos(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .value())
            .margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(
            (3 * (2 * x + 1) * sleipnir::cos(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .value())
            .margin(1e-15));

  g = sleipnir::Gradient(3 * sleipnir::atan2(sleipnir::sin(y), 2 * x + 1), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(
            (3 * -2 * sleipnir::sin(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .value())
            .margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(
            (3 * -2 * sleipnir::sin(y) /
             ((2 * x + 1) * (2 * x + 1) + sleipnir::sin(y) * sleipnir::sin(y)))
                .value())
            .margin(1e-15));
}

TEST_CASE("Gradient - std::hypot()", "[Gradient]") {
  sleipnir::Variable x;
  sleipnir::Variable y;

  // Testing hypot function on (var, double)
  x.set_value(1.8);
  y.set_value(1.5);
  CHECK(sleipnir::hypot(x, 2.0).value() == std::hypot(x.value(), 2.0));

  auto g = sleipnir::Gradient(sleipnir::hypot(x, 2.0), x);
  CHECK(g.get().value().coeff(0) == (x / std::hypot(x.value(), 2.0)).value());
  CHECK(g.value().coeff(0) == (x / std::hypot(x.value(), 2.0)).value());

  // Testing hypot function on (double, var)
  CHECK(sleipnir::hypot(2.0, y).value() == std::hypot(2.0, y.value()));

  g = sleipnir::Gradient(sleipnir::hypot(2.0, y), y);
  CHECK(g.get().value().coeff(0) == (y / std::hypot(2.0, y.value())).value());
  CHECK(g.value().coeff(0) == (y / std::hypot(2.0, y.value())).value());

  // Testing hypot function on (var, var)
  x.set_value(1.3);
  y.set_value(2.3);
  CHECK(sleipnir::hypot(x, y).value() == std::hypot(x.value(), y.value()));

  g = sleipnir::Gradient(sleipnir::hypot(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        (x / std::hypot(x.value(), y.value())).value());
  CHECK(g.value().coeff(0) == (x / std::hypot(x.value(), y.value())).value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        (y / std::hypot(x.value(), y.value())).value());
  CHECK(g.value().coeff(0) == (y / std::hypot(x.value(), y.value())).value());

  // Testing hypot function on (expr, expr)
  x.set_value(1.3);
  y.set_value(2.3);
  CHECK(sleipnir::hypot(2.0 * x, 3.0 * y).value() ==
        std::hypot(2.0 * x.value(), 3.0 * y.value()));

  g = sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), x);
  CHECK(g.get().value().coeff(0) ==
        (4.0 * x / std::hypot(2.0 * x.value(), 3.0 * y.value())).value());
  CHECK(g.value().coeff(0) ==
        (4.0 * x / std::hypot(2.0 * x.value(), 3.0 * y.value())).value());

  g = sleipnir::Gradient(sleipnir::hypot(2.0 * x, 3.0 * y), y);
  CHECK(g.get().value().coeff(0) ==
        (9.0 * y / std::hypot(2.0 * x.value(), 3.0 * y.value())).value());
  CHECK(g.value().coeff(0) ==
        (9.0 * y / std::hypot(2.0 * x.value(), 3.0 * y.value())).value());

  // Testing hypot function on (var, var, var)
  sleipnir::Variable z;
  x.set_value(1.3);
  y.set_value(2.3);
  z.set_value(3.3);
  CHECK(sleipnir::hypot(x, y, z).value() ==
        std::hypot(x.value(), y.value(), z.value()));

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), x);
  CHECK(g.get().value().coeff(0) ==
        (x / std::hypot(x.value(), y.value(), z.value())).value());
  CHECK(g.value().coeff(0) ==
        (x / std::hypot(x.value(), y.value(), z.value())).value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), y);
  CHECK(g.get().value().coeff(0) ==
        (y / std::hypot(x.value(), y.value(), z.value())).value());
  CHECK(g.value().coeff(0) ==
        (y / std::hypot(x.value(), y.value(), z.value())).value());

  g = sleipnir::Gradient(sleipnir::hypot(x, y, z), z);
  CHECK(g.get().value().coeff(0) ==
        (z / std::hypot(x.value(), y.value(), z.value())).value());
  CHECK(g.value().coeff(0) ==
        (z / std::hypot(x.value(), y.value(), z.value())).value());
}

TEST_CASE("Gradient - Miscellaneous", "[Gradient]") {
  sleipnir::Variable x;

  // dx/dx
  x.set_value(3.0);
  CHECK(sleipnir::abs(x).value() == std::abs(x.value()));

  auto g = sleipnir::Gradient(x, x);
  CHECK(g.get().value().coeff(0) == 1.0);
  CHECK(g.value().coeff(0) == 1.0);

  // std::erf(x)
  x.set_value(0.5);
  CHECK(sleipnir::erf(x).value() == std::erf(x.value()));

  g = sleipnir::Gradient(sleipnir::erf(x), x);
  CHECK(g.get().value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.value() * x.value()));
  CHECK(g.value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.value() * x.value()));
}

TEST_CASE("Gradient - Variable reuse", "[Gradient]") {
  sleipnir::Variable a;
  a.set_value(10);

  sleipnir::Variable b;
  b.set_value(20);

  sleipnir::Variable x = a * b;

  auto g = sleipnir::Gradient(x, a);

  CHECK(g.get().value().coeff(0) == 20.0);
  CHECK(g.value().coeff(0) == 20.0);

  b.set_value(10);
  CHECK(g.get().value().coeff(0) == 10.0);
  CHECK(g.value().coeff(0) == 10.0);
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
  x.set_value(1.0);
  CHECK(sleipnir::sign(x).value() == sign(x.value()));

  auto g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);

  // sgn(-1.0)
  x.set_value(-1.0);
  CHECK(sleipnir::sign(x).value() == sign(x.value()));

  g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);

  // sgn(0.0)
  x.set_value(0.0);
  CHECK(sleipnir::sign(x).value() == sign(x.value()));

  g = sleipnir::Gradient(sleipnir::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);
}
