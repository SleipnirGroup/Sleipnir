// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/optimization/Constraints.hpp"
#include "sleipnir/optimization/SolverConfig.hpp"
#include "sleipnir/optimization/SolverStatus.hpp"

namespace sleipnir {

/**
 * This class allows the user to pose a constrained nonlinear optimization
 * problem in natural mathematical notation and solve it.
 *
 * This class supports problems of the form:
@verbatim
      minₓ f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
@endverbatim
 *
 * where f(x) is the scalar cost function, x is the vector of decision variables
 * (variables the solver can tweak to minimize the cost function), cᵢ(x) are the
 * inequality constraints, and cₑ(x) are the equality constraints. Constraints
 * are equations or inequalities of the decision variables that constrain what
 * values the solver is allowed to use when searching for an optimal solution.
 *
 * The nice thing about this class is users don't have to put their system in
 * the form shown above manually; they can write it in natural mathematical form
 * and it'll be converted for them. We'll cover some examples next.
 *
 * ## Double integrator minimum time
 *
 * A system with position and velocity states and an acceleration input is an
 * example of a double integrator. We want to go from 0 m at rest to 10 m at
 * rest in the minimum time while obeying the velocity limit (-1, 1) and the
 * acceleration limit (-1, 1).
 *
 * The model for our double integrator is ẍ=u where x is the vector [position;
 * velocity] and u is the acceleration. The velocity constraints are -1 ≤ x(1)
 * ≤ 1 and the acceleration constraints are -1 ≤ u ≤ 1.
 *
 * ### Initializing a problem instance
 *
 * First, we need to make a problem instance.
 * @code{.cpp}
 * #include <Eigen/Core>
 * #include <sleipnir/optimization/OptimizationProblem.hpp>
 *
 * int main() {
 *   constexpr auto T = 5s;
 *   constexpr auto dt = 5ms;
 *   constexpr int N = T / dt;
 *
 *   sleipnir::OptimizationProblem problem;
 * @endcode
 *
 * ### Creating decision variables
 *
 * First, we need to make decision variables for our state and input.
 * @code{.cpp}
 *   // 2x1 state vector with N + 1 timesteps (includes last state)
 *   auto X = problem.DecisionVariable(2, N + 1);
 *
 *   // 1x1 input vector with N timesteps (input at last state doesn't matter)
 *   auto U = problem.DecisionVariable(1, N);
 * @endcode
 * By convention, we use capital letters for the variables to designate
 * matrices.
 *
 * ### Applying constraints
 *
 * Now, we need to apply dynamics constraints between timesteps.
 * @code{.cpp}
 * // Kinematics constraint assuming constant acceleration between timesteps
 * for (int k = 0; k < N; ++k) {
 *   constexpr double t = std::chrono::duration<double>(dt).count();
 *   auto p_k1 = X(0, k + 1);
 *   auto v_k1 = X(1, k + 1);
 *   auto p_k = X(0, k);
 *   auto v_k = X(1, k);
 *   auto a_k = U(0, k);
 *
 *   // pₖ₊₁ = pₖ + vₖt
 *   problem.SubjectTo(p_k1 == p_k + v_k * t);
 *
 *   // vₖ₊₁ = vₖ + aₖt
 *   problem.SubjectTo(v_k1 == v_k + a_k * t);
 * }
 * @endcode
 *
 * Next, we'll apply the state and input constraints.
 * @code{.cpp}
 * // Start and end at rest
 * problem.SubjectTo(X.Col(0) == Eigen::Matrix<double, 2, 1>{{0.0}, {0.0}});
 * problem.SubjectTo(
 *   X.Col(N + 1) == Eigen::Matrix<double, 2, 1>{{10.0}, {0.0}});
 *
 * // Limit velocity
 * problem.SubjectTo(-1 <= X.Row(1));
 * problem.SubjectTo(X.Row(1) <= 1);
 *
 * // Limit acceleration
 * problem.SubjectTo(-1 <= U);
 * problem.SubjectTo(U <= 1);
 * @endcode
 *
 * ### Specifying a cost function
 *
 * Next, we'll create a cost function for minimizing position error.
 * @code{.cpp}
 * // Cost function - minimize position error
 * sleipnir::VariableMatrix J = 0.0;
 * for (int k = 0; k < N + 1; ++k) {
 *   J += sleipnir::pow(10.0 - X(0, k), 2);
 * }
 * problem.Minimize(J);
 * @endcode
 * The cost function passed to Minimize() should produce a scalar output.
 *
 * ### Solving the problem
 *
 * Now we can solve the problem.
 * @code{.cpp}
 * problem.Solve();
 * @endcode
 *
 * The solver will find the decision variable values that minimize the cost
 * function while satisfying the constraints.
 *
 * ### Accessing the solution
 *
 * You can obtain the solution by querying the values of the variables like so.
 * @code{.cpp}
 * double position = X.Value(0, 0);
 * double velocity = X.Value(1, 0);
 * double acceleration = U.Value(0);
 * @endcode
 *
 * ### Other applications
 *
 * In retrospect, the solution here seems obvious: if you want to reach the
 * desired position in the minimum time, you just apply max input to move toward
 * it, then stop applying input once you get there. Optimization problems can
 * get more complex than this though. In fact, we can use this same framework to
 * design optimal trajectories for a drivetrain while satisfying dynamics
 * constraints, avoiding obstacles, and driving through points of interest.
 *
 * ## Optimizing the problem formulation
 *
 * Cost functions and constraints can have the following orders:
 *
 * <ul>
 *   <li>none (i.e., there is no cost function or are no constraints)</li>
 *   <li>constant</li>
 *   <li>linear</li>
 *   <li>quadratic</li>
 *   <li>nonlinear</li>
 * </ul>
 *
 * For nonlinear problems, the solver calculates the Hessian of the cost
 * function and the Jacobians of the constraints at each iteration. However,
 * problems with lower order cost functions and constraints can be solved
 * faster. For example, the following only need to be computed once because
 * they're constant:
 *
 * <ul>
 *   <li>the Hessian of a quadratic or lower cost function</li>
 *   <li>the Jacobian of linear or lower constraints</li>
 * </ul>
 *
 * A problem is constant if:
 *
 * <ul>
 *   <li>the cost function is constant or lower</li>
 *   <li>the equality constraints are constant or lower</li>
 *   <li>the inequality constraints are constant or lower</li>
 * </ul>
 *
 * A problem is linear if:
 *
 * <ul>
 *   <li>the cost function is linear</li>
 *   <li>the equality constraints are linear or lower</li>
 *   <li>the inequality constraints are linear or lower</li>
 * </ul>
 *
 * A problem is quadratic if:
 *
 * <ul>
 *   <li>the cost function is quadratic</li>
 *   <li>the equality constraints are linear or lower</li>
 *   <li>the inequality constraints are linear or lower</li>
 * </ul>
 *
 * All other problems are nonlinear.
 */
class SLEIPNIR_DLLEXPORT OptimizationProblem {
 public:
  /**
   * Construct the optimization problem.
   */
  OptimizationProblem() noexcept;

  /**
   * Create a matrix of decision variables in the optimization problem.
   *
   * @param rows Number of matrix rows.
   * @param cols Number of matrix columns.
   */
  [[nodiscard]] VariableMatrix DecisionVariable(int rows = 1, int cols = 1);

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize. It must return a 1x1 matrix.
   */
  void Minimize(const VariableMatrix& cost);

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize. It must return a 1x1 matrix.
   */
  void Minimize(VariableMatrix&& cost);

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize. It must return a 1x1
   *                  matrix.
   */
  void Maximize(const VariableMatrix& objective);

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize. It must return a 1x1
   *                  matrix.
   */
  void Maximize(VariableMatrix&& objective);

  /**
   * Tells the solver to solve the problem while satisfying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(EqualityConstraints&& constraint);

  /**
   * Tells the solver to solve the problem while satisfying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(InequalityConstraints&& constraint);

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   *
   * @param config Configuration options for the solver.
   */
  SolverStatus Solve(const SolverConfig& config = kDefaultConfig);

 private:
  // GCC incorrectly applies C++14 rules for const static data members, so an
  // initializer is required here.
  //
  // https://stackoverflow.com/a/50639754
  static constexpr SolverConfig kDefaultConfig{};

  // Decision variables, which are the root of the problem's expression tree
  std::vector<Variable> m_decisionVariables;

  // Cost function: f(x)
  std::optional<Variable> m_f;

  // Equality constraints: cₑ(x) = 0
  std::vector<Variable> m_equalityConstraints;

  // Inequality constraints: cᵢ(x) ≥ 0
  std::vector<Variable> m_inequalityConstraints;

  SolverConfig m_config;

  /**
  Find the optimal solution to the nonlinear program using an interior-point
  solver.

  A nonlinear program has the form:

  @verbatim
       min_x f(x)
  subject to cₑ(x) = 0
             cᵢ(x) ≥ 0
  @endverbatim

  where f(x) is the cost function, cₑ(x) are the equality constraints, and cᵢ(x)
  are the inequality constraints.

  @param[in] initialGuess The initial guess.
  @param[out] status The solver status.
  */
  Eigen::VectorXd InteriorPoint(
      const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
      SolverStatus* status);
};

}  // namespace sleipnir
