// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/problem.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point.hpp"
#include "sleipnir/optimization/solver/newton.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/setup_profiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "util/print_diagnostics.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/print.hpp"
#endif

namespace slp {

ExitStatus Problem::solve(const Options& options) {
  // Create the initial value column vector
  Eigen::VectorXd x{m_decision_variables.size()};
  for (size_t i = 0; i < m_decision_variables.size(); ++i) {
    x[i] = m_decision_variables[i].value();
  }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  if (options.diagnostics) {
    // Print possible exit conditions
    slp::println("User-configured exit conditions:");
    slp::println("  ↳ error below {}", options.tolerance);
    if (!m_callbacks.empty()) {
      slp::println("  ↳ iteration callback requested stop");
    }
    if (std::isfinite(options.max_iterations)) {
      slp::println("  ↳ executed {} iterations", options.max_iterations);
    }
    if (std::isfinite(options.timeout.count())) {
      slp::println("  ↳ {} elapsed", options.timeout);
    }

    if (m_decision_variables.size() == 1) {
      slp::println("\n1 decision variable");
    } else {
      slp::println("\n{} decision variables", m_decision_variables.size());
    }

    auto print_constraint_types =
        [](const small_vector<Variable>& constraints) {
          std::array<size_t, 5> type_counts{};
          for (const auto& constraint : constraints) {
            ++type_counts[std::to_underlying(constraint.type())];
          }
          for (const auto& [count, name] : std::views::zip(
                   type_counts, std::array{"empty", "constant", "linear",
                                           "quadratic", "nonlinear"})) {
            if (count > 0) {
              slp::println("  ↳ {} {}", count, name);
            }
          }
        };

    // Print constraint types
    if (m_equality_constraints.size() == 1) {
      slp::println("1 equality constraint");
    } else {
      slp::println("{} equality constraints", m_equality_constraints.size());
    }
    print_constraint_types(m_equality_constraints);
    if (m_inequality_constraints.size() == 1) {
      slp::println("1 inequality constraint");
    } else {
      slp::println("{} inequality constraints",
                   m_inequality_constraints.size());
    }
    print_constraint_types(m_inequality_constraints);
  }

  auto print_chosen_solver =
      [](std::string_view solver_name, const ExpressionType& f_type,
         const ExpressionType& c_e_type, const ExpressionType& c_i_type) {
        constexpr std::array types{"no", "constant", "linear", "quadratic",
                                   "nonlinear"};

        slp::println("\nUsing {} solver due to:", solver_name);
        slp::println("  ↳ {} cost function", types[std::to_underlying(f_type)]);
        slp::println("  ↳ {} equality constraints",
                     types[std::to_underlying(c_e_type)]);
        slp::println("  ↳ {} inequality constraints",
                     types[std::to_underlying(c_i_type)]);
        slp::println("");
      };
#endif

  // Get the highest order constraint expression types
  auto f_type = cost_function_type();
  auto c_e_type = equality_constraint_type();
  auto c_i_type = inequality_constraint_type();

  // If the problem is empty or constant, there's nothing to do
  if (f_type <= ExpressionType::CONSTANT &&
      c_e_type <= ExpressionType::CONSTANT &&
      c_i_type <= ExpressionType::CONSTANT) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_chosen_solver("no-op", f_type, c_e_type, c_i_type);
    }
#endif
    return ExitStatus::SUCCESS;
  }

  small_vector<SetupProfiler> autodiff_setup_profilers;
  autodiff_setup_profilers.emplace_back("setup").start();

  VariableMatrix x_ad{m_decision_variables};

  // Set up cost function
  Variable f = m_f.value_or(0.0);

  // Set up gradient autodiff
  autodiff_setup_profilers.emplace_back("  ↳ ∇f(x)").start();
  Gradient g{f, x_ad};
  autodiff_setup_profilers.back().stop();

  // Solve the optimization problem
  ExitStatus status;
  if (m_equality_constraints.empty() && m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_chosen_solver("Newton", f_type, c_e_type, c_i_type);
    }
#endif

    // Set up Lagrangian Hessian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∇²ₓₓL").start();
    Hessian<Eigen::Lower> H{f, x_ad};
    autodiff_setup_profilers.back().stop();

    autodiff_setup_profilers[0].stop();

    NewtonMatrixCallbacks matrix_callbacks{
        [&](const Eigen::VectorXd& x) -> double {
          x_ad.set_value(x);
          return f.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseVector<double> {
          x_ad.set_value(x);
          return g.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          return H.value();
        }};

    SetupProfiler total_solve_profiler{"solve"};
    {
      ScopedProfiler scoped_prof{total_solve_profiler};
      status = newton(matrix_callbacks, m_callbacks, options, x);
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      small_vector<SolveProfiler> autodiff_solve_profilers;

      // Append gradient profilers
      autodiff_solve_profilers.push_back(g.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇f(x)";
      for (const auto& profiler : g.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append Hessian profilers
      autodiff_solve_profilers.push_back(H.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇²ₓₓL";
      for (const auto& profiler : H.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      print_autodiff_setup_diagnostics(autodiff_setup_profilers);
      print_autodiff_solve_diagnostics(total_solve_profiler,
                                       autodiff_solve_profilers);
    }
#endif
  } else if (m_inequality_constraints.empty()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_chosen_solver("SQP", f_type, c_e_type, c_i_type);
    }
#endif

    VariableMatrix c_e_ad{m_equality_constraints};

    // Set up equality constraint Jacobian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∂cₑ/∂x").start();
    Jacobian A_e{c_e_ad, x_ad};
    autodiff_setup_profilers.back().stop();

    VariableMatrix y_ad(m_equality_constraints.size());
    Variable L = f - (y_ad.T() * c_e_ad)[0];

    // Set up Lagrangian Hessian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∇²ₓₓL").start();
    Hessian<Eigen::Lower> H{L, x_ad};
    autodiff_setup_profilers.back().stop();

    autodiff_setup_profilers[0].stop();

    SQPMatrixCallbacks matrix_callbacks{
        [&](const Eigen::VectorXd& x) -> double {
          x_ad.set_value(x);
          return f.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseVector<double> {
          x_ad.set_value(x);
          return g.value();
        },
        [&](const Eigen::VectorXd& x,
            const Eigen::VectorXd& y) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          y_ad.set_value(y);
          return H.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
          x_ad.set_value(x);
          return c_e_ad.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          return A_e.value();
        }};

    SetupProfiler total_solve_profiler{"solve"};
    {
      ScopedProfiler scoped_prof{total_solve_profiler};
      status = sqp(matrix_callbacks, m_callbacks, options, x);
    }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      small_vector<SolveProfiler> autodiff_solve_profilers;

      // Append gradient profilers
      autodiff_solve_profilers.push_back(g.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇f(x)";
      for (const auto& profiler : g.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append Hessian profilers
      autodiff_solve_profilers.push_back(H.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇²ₓₓL";
      for (const auto& profiler : H.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append equality constraint Jacobian profilers
      autodiff_solve_profilers.push_back(A_e.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∂cₑ/∂x";
      for (const auto& profiler : A_e.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      print_autodiff_setup_diagnostics(autodiff_setup_profilers);
      print_autodiff_solve_diagnostics(total_solve_profiler,
                                       autodiff_solve_profilers);
    }
#endif
  } else {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_chosen_solver("IPM", f_type, c_e_type, c_i_type);
    }
#endif

    VariableMatrix c_e_ad{m_equality_constraints};
    VariableMatrix c_i_ad{m_inequality_constraints};

    // Set up equality constraint Jacobian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∂cₑ/∂x").start();
    Jacobian A_e{c_e_ad, x_ad};
    autodiff_setup_profilers.back().stop();

    // Set up inequality constraint Jacobian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∂cᵢ/∂x").start();
    Jacobian A_i{c_i_ad, x_ad};
    autodiff_setup_profilers.back().stop();

    VariableMatrix y_ad(m_equality_constraints.size());
    VariableMatrix z_ad(m_inequality_constraints.size());
    Variable L = f - (y_ad.T() * c_e_ad)[0] - (z_ad.T() * c_i_ad)[0];

    // Set up Lagrangian Hessian autodiff
    autodiff_setup_profilers.emplace_back("  ↳ ∇²ₓₓL").start();
    Hessian<Eigen::Lower> H{L, x_ad};
    autodiff_setup_profilers.back().stop();

    autodiff_setup_profilers[0].stop();

    InteriorPointMatrixCallbacks matrix_callbacks{
        [&](const Eigen::VectorXd& x) -> double {
          x_ad.set_value(x);
          return f.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseVector<double> {
          x_ad.set_value(x);
          return g.value();
        },
        [&](const Eigen::VectorXd& x, const Eigen::VectorXd& y,
            const Eigen::VectorXd& z) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          y_ad.set_value(y);
          z_ad.set_value(z);
          return H.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
          x_ad.set_value(x);
          return c_e_ad.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          return A_e.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
          x_ad.set_value(x);
          return c_i_ad.value();
        },
        [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
          x_ad.set_value(x);
          return A_i.value();
        }};

    SetupProfiler total_solve_profiler{"solve"};
    {
      ScopedProfiler scoped_prof{total_solve_profiler};
      status = interior_point(matrix_callbacks, m_callbacks, options, x);
    }

    // Prints final diagnostics when the solver exits
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      small_vector<SolveProfiler> autodiff_solve_profilers;

      // Append gradient profilers
      autodiff_solve_profilers.push_back(g.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇f(x)";
      for (const auto& profiler : g.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append Hessian profilers
      autodiff_solve_profilers.push_back(H.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∇²ₓₓL";
      for (const auto& profiler : H.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append equality constraint Jacobian profilers
      autodiff_solve_profilers.push_back(A_e.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∂cₑ/∂x";
      for (const auto& profiler : A_e.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      // Append inequality constraint Jacobian profilers
      autodiff_solve_profilers.push_back(A_i.get_profilers()[0]);
      autodiff_solve_profilers.back().name = "  ↳ ∂cᵢ/∂x";
      for (const auto& profiler : A_i.get_profilers() | std::views::drop(1)) {
        autodiff_solve_profilers.push_back(profiler);
      }

      print_autodiff_setup_diagnostics(autodiff_setup_profilers);
      print_autodiff_solve_diagnostics(total_solve_profiler,
                                       autodiff_solve_profilers);
    }
#endif
  }

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  if (options.diagnostics) {
    slp::println("\nExit: {}", to_message(status));
  }
#endif

  // Assign the solution to the original Variable instances
  VariableMatrix{m_decision_variables}.set_value(x);

  return status;
}

}  // namespace slp
