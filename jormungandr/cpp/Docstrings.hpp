/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_sleipnir_Block =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_sleipnir_Block_2 =
R"doc(Assemble a VariableMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of blocks.)doc";

static const char *__doc_sleipnir_CwiseReduce =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Parameter ``lhs``:
    The left-hand side of the binary operator.

Parameter ``rhs``:
    The right-hand side of the binary operator.

Parameter ``binaryOp``:
    The binary operator to use for the reduce operation.)doc";

static const char *__doc_sleipnir_DynamicsType = R"doc(Enum describing a type of system dynamics constraints.)doc";

static const char *__doc_sleipnir_DynamicsType_kDiscrete = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

static const char *__doc_sleipnir_DynamicsType_kExplicitODE = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

static const char *__doc_sleipnir_EqualityConstraints = R"doc(A vector of equality constraints of the form cₑ(x) = 0.)doc";

static const char *__doc_sleipnir_EqualityConstraints_EqualityConstraints =
R"doc(Constructs an equality constraint from a left and right side.

The standard form for equality constraints is c(x) = 0. This function
takes a constraint of the form lhs = rhs and converts it to lhs - rhs
= 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_EqualityConstraints_constraints = R"doc(A vector of scalar equality constraints.)doc";

static const char *__doc_sleipnir_EqualityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_sleipnir_ExpressionType =
R"doc(Expression type.

Used for autodiff caching.)doc";

static const char *__doc_sleipnir_ExpressionType_kConstant = R"doc(The expression is a constant.)doc";

static const char *__doc_sleipnir_ExpressionType_kLinear = R"doc(The expression is composed of linear and lower-order operators.)doc";

static const char *__doc_sleipnir_ExpressionType_kNone = R"doc(There is no expression.)doc";

static const char *__doc_sleipnir_ExpressionType_kNonlinear = R"doc(The expression is composed of nonlinear and lower-order operators.)doc";

static const char *__doc_sleipnir_ExpressionType_kQuadratic = R"doc(The expression is composed of quadratic and lower-order operators.)doc";

static const char *__doc_sleipnir_Gradient =
R"doc(This class calculates the gradient of a a variable with respect to a
vector of variables.

The gradient is only recomputed if the variable expression is
quadratic or higher order.)doc";

static const char *__doc_sleipnir_Gradient_Get =
R"doc(Returns the gradient as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Gradient_GetProfiler = R"doc(Returns the profiler.)doc";

static const char *__doc_sleipnir_Gradient_Gradient =
R"doc(Constructs a Gradient object.

Parameter ``variable``:
    Variable of which to compute the gradient.

Parameter ``wrt``:
    Variable with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_Gradient_Gradient_2 =
R"doc(Constructs a Gradient object.

Parameter ``variable``:
    Variable of which to compute the gradient.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_Gradient_Update = R"doc(Updates the value of the variable.)doc";

static const char *__doc_sleipnir_Gradient_Value = R"doc(Evaluates the gradient at wrt's value.)doc";

static const char *__doc_sleipnir_Gradient_m_g = R"doc()doc";

static const char *__doc_sleipnir_Gradient_m_jacobian = R"doc()doc";

static const char *__doc_sleipnir_Hessian =
R"doc(This class calculates the Hessian of a variable with respect to a
vector of variables.

The gradient tree is cached so subsequent Hessian calculations are
faster, and the Hessian is only recomputed if the variable expression
is nonlinear.)doc";

static const char *__doc_sleipnir_Hessian_Get =
R"doc(Returns the Hessian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Hessian_GetProfiler = R"doc(Returns the profiler.)doc";

static const char *__doc_sleipnir_Hessian_Hessian =
R"doc(Constructs a Hessian object.

Parameter ``variable``:
    Variable of which to compute the Hessian.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the Hessian.)doc";

static const char *__doc_sleipnir_Hessian_Update = R"doc(Updates the values of the gradient tree.)doc";

static const char *__doc_sleipnir_Hessian_Value = R"doc(Evaluates the Hessian at wrt's value.)doc";

static const char *__doc_sleipnir_Hessian_m_jacobian = R"doc()doc";

static const char *__doc_sleipnir_InequalityConstraints = R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.)doc";

static const char *__doc_sleipnir_InequalityConstraints_InequalityConstraints =
R"doc(Constructs an inequality constraint from a left and right side.

The standard form for inequality constraints is c(x) ≥ 0. This
function takes a constraints of the form lhs ≥ rhs and converts it to
lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_InequalityConstraints_constraints = R"doc(A vector of scalar inequality constraints.)doc";

static const char *__doc_sleipnir_InequalityConstraints_operator_bool = R"doc(Implicit conversion operator to bool.)doc";

static const char *__doc_sleipnir_Jacobian =
R"doc(This class calculates the Jacobian of a vector of variables with
respect to a vector of variables.

The Jacobian is only recomputed if the variable expression is
quadratic or higher order.)doc";

static const char *__doc_sleipnir_Jacobian_2 = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_Get =
R"doc(Returns the Jacobian as a VariableMatrix.

This is useful when constructing optimization problems with
derivatives in them.)doc";

static const char *__doc_sleipnir_Jacobian_GetProfiler = R"doc(Returns the profiler.)doc";

static const char *__doc_sleipnir_Jacobian_Jacobian =
R"doc(Constructs a Jacobian object.

Parameter ``variables``:
    Vector of variables of which to compute the Jacobian.

Parameter ``wrt``:
    Vector of variables with respect to which to compute the Jacobian.)doc";

static const char *__doc_sleipnir_Jacobian_Update = R"doc(Updates the values of the variables.)doc";

static const char *__doc_sleipnir_Jacobian_Value = R"doc(Evaluates the Jacobian at wrt's value.)doc";

static const char *__doc_sleipnir_Jacobian_m_J = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_cachedTriplets = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_graphs = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_nonlinearRows = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_profiler = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_variables = R"doc()doc";

static const char *__doc_sleipnir_Jacobian_m_wrt = R"doc()doc";

static const char *__doc_sleipnir_MakeConstraints =
R"doc(Make a list of constraints.

The standard form for equality constraints is c(x) = 0, and the
standard form for inequality constraints is c(x) ≥ 0. This function
takes constraints of the form lhs = rhs or lhs ≥ rhs and converts them
to lhs - rhs = 0 or lhs - rhs ≥ 0.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Right-hand side.)doc";

static const char *__doc_sleipnir_OCPSolver =
R"doc(This class allows the user to pose and solve a constrained optimal
control problem (OCP) in a variety of ways.

The system is transcripted by one of three methods (direct
transcription, direct collocation, or single-shooting) and additional
constraints can be added.

In direct transcription, each state is a decision variable constrained
to the integrated dynamics of the previous state. In direct
collocation, the trajectory is modeled as a series of cubic
polynomials where the centerpoint slope is constrained. In single-
shooting, states depend explicitly as a function of all previous
states and all previous inputs.

Explicit ODEs are integrated using RK4.

For explicit ODEs, the function must be in the form dx/dt = f(t, x,
u). For discrete state transition functions, the function must be in
the form xₖ₊₁ = f(t, xₖ, uₖ).

Direct collocation requires an explicit ODE. Direct transcription and
single-shooting can use either an ODE or state transition function.

https://underactuated.mit.edu/trajopt.html goes into more detail on
each transcription method.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainAlways =
R"doc(Set the constraint evaluation function. This function is called
`numSteps+1` times, with the corresponding state and input
VariableMatrices.

Parameter ``constraintFunction``:
    the constraint function.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainDirectCollocation = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainDirectTranscription = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainFinalState =
R"doc(Utility function to constrain the final state.

Parameter ``finalState``:
    the final state to constrain to.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainInitialState =
R"doc(Utility function to constrain the initial state.

Parameter ``initialState``:
    the initial state to constrain to.)doc";

static const char *__doc_sleipnir_OCPSolver_ConstrainSingleShooting = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_DT =
R"doc(Get the timestep variables. After the problem is solved, this will
contain the timesteps corresponding to the optimized trajectory.

Shaped 1x(numSteps+1), although the last timestep is unused in the
trajectory.

Returns:
    The timestep variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_FinalState =
R"doc(Convenience function to get the final state in the trajectory.

Returns:
    The final state of the trajectory.)doc";

static const char *__doc_sleipnir_OCPSolver_InitialState =
R"doc(Convenience function to get the initial state in the trajectory.

Returns:
    The initial state of the trajectory.)doc";

static const char *__doc_sleipnir_OCPSolver_OCPSolver =
R"doc(Build an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Parameter ``numStates``:
    The number of system states.

Parameter ``numInputs``:
    The number of system inputs.

Parameter ``dt``:
    The timestep for fixed-step integration.

Parameter ``numSteps``:
    The number of control points.

Parameter ``dynamics``:
    The system evolution function, either an explicit ODE or a
    discrete state transition function.

Parameter ``dynamicsType``:
    The type of system evolution function.

Parameter ``timestepMethod``:
    The timestep method.

Parameter ``method``:
    The transcription method.)doc";

static const char *__doc_sleipnir_OCPSolver_SetLowerInputBound =
R"doc(Convenience function to set a lower bound on the input.

Parameter ``lowerBound``:
    The lower bound that inputs must always be above. Must be shaped
    (numInputs)x1.)doc";

static const char *__doc_sleipnir_OCPSolver_SetMaxTimestep =
R"doc(Convenience function to set an upper bound on the timestep.

Parameter ``maxTimestep``:
    The maximum timestep.)doc";

static const char *__doc_sleipnir_OCPSolver_SetMinTimestep =
R"doc(Convenience function to set a lower bound on the timestep.

Parameter ``minTimestep``:
    The minimum timestep.)doc";

static const char *__doc_sleipnir_OCPSolver_SetUpperInputBound =
R"doc(Convenience function to set an upper bound on the input.

Parameter ``upperBound``:
    The upper bound that inputs must always be below. Must be shaped
    (numInputs)x1.)doc";

static const char *__doc_sleipnir_OCPSolver_U =
R"doc(Get the input variables. After the problem is solved, this will
contain the inputs corresponding to the optimized trajectory.

Shaped (numInputs)x(numSteps+1), although the last input step is
unused in the trajectory.

Returns:
    The input variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_X =
R"doc(Get the state variables. After the problem is solved, this will
contain the optimized trajectory.

Shaped (numStates)x(numSteps+1).

Returns:
    The state variable matrix.)doc";

static const char *__doc_sleipnir_OCPSolver_m_DT = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_U = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_X = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dt = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dynamicsFunction = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_dynamicsType = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numInputs = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numStates = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_numSteps = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_timestepMethod = R"doc()doc";

static const char *__doc_sleipnir_OCPSolver_m_transcriptionMethod = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem =
R"doc(This class allows the user to pose a constrained nonlinear
optimization problem in natural mathematical notation and solve it.

This class supports problems of the form: @verbatim minₓ f(x) subject
to cₑ(x) = 0 cᵢ(x) ≥ 0 @endverbatim

where f(x) is the scalar cost function, x is the vector of decision
variables (variables the solver can tweak to minimize the cost
function), cᵢ(x) are the inequality constraints, and cₑ(x) are the
equality constraints. Constraints are equations or inequalities of the
decision variables that constrain what values the solver is allowed to
use when searching for an optimal solution.

The nice thing about this class is users don't have to put their
system in the form shown above manually; they can write it in natural
mathematical form and it'll be converted for them. We'll cover some
examples next.

## Double integrator minimum time

A system with position and velocity states and an acceleration input
is an example of a double integrator. We want to go from 0 m at rest
to 10 m at rest in the minimum time while obeying the velocity limit
(-1, 1) and the acceleration limit (-1, 1).

The model for our double integrator is ẍ=u where x is the vector
[position; velocity] and u is the acceleration. The velocity
constraints are -1 ≤ x(1) ≤ 1 and the acceleration constraints are -1
≤ u ≤ 1.

### Initializing a problem instance

First, we need to make a problem instance.

```
{.cpp}
 #include <Eigen/Core>
 #include <sleipnir/optimization/OptimizationProblem.hpp>

 int main() {
   constexpr auto T = 5s;
   constexpr auto dt = 5ms;
   constexpr int N = T / dt;

   sleipnir::OptimizationProblem problem;
```

### Creating decision variables

First, we need to make decision variables for our state and input.

```
{.cpp}
   // 2x1 state vector with N + 1 timesteps (includes last state)
   auto X = problem.DecisionVariable(2, N + 1);

   // 1x1 input vector with N timesteps (input at last state doesn't matter)
   auto U = problem.DecisionVariable(1, N);
```

By convention, we use capital letters for the variables to designate
matrices.

### Applying constraints

Now, we need to apply dynamics constraints between timesteps.

```
{.cpp}
 // Kinematics constraint assuming constant acceleration between timesteps
 for (int k = 0; k < N; ++k) {
   constexpr double t = std::chrono::duration<double>(dt).count();
   auto p_k1 = X(0, k + 1);
   auto v_k1 = X(1, k + 1);
   auto p_k = X(0, k);
   auto v_k = X(1, k);
   auto a_k = U(0, k);

   // pₖ₊₁ = pₖ + vₖt
   problem.SubjectTo(p_k1 == p_k + v_k * t);

   // vₖ₊₁ = vₖ + aₖt
   problem.SubjectTo(v_k1 == v_k + a_k * t);
 }
```

Next, we'll apply the state and input constraints.

```
{.cpp}
 // Start and end at rest
 problem.SubjectTo(X.Col(0) == Eigen::Matrix<double, 2, 1>{{0.0}, {0.0}});
 problem.SubjectTo(
   X.Col(N + 1) == Eigen::Matrix<double, 2, 1>{{10.0}, {0.0}});

 // Limit velocity
 problem.SubjectTo(-1 <= X.Row(1));
 problem.SubjectTo(X.Row(1) <= 1);

 // Limit acceleration
 problem.SubjectTo(-1 <= U);
 problem.SubjectTo(U <= 1);
```

### Specifying a cost function

Next, we'll create a cost function for minimizing position error.

```
{.cpp}
 // Cost function - minimize position error
 sleipnir::Variable J = 0.0;
 for (int k = 0; k < N + 1; ++k) {
   J += sleipnir::pow(10.0 - X(0, k), 2);
 }
 problem.Minimize(J);
```

The cost function passed to Minimize() should produce a scalar output.

### Solving the problem

Now we can solve the problem.

```
{.cpp}
 problem.Solve();
```

The solver will find the decision variable values that minimize the
cost function while satisfying the constraints.

### Accessing the solution

You can obtain the solution by querying the values of the variables
like so.

```
{.cpp}
 double position = X.Value(0, 0);
 double velocity = X.Value(1, 0);
 double acceleration = U.Value(0);
```

### Other applications

In retrospect, the solution here seems obvious: if you want to reach
the desired position in the minimum time, you just apply positive max
input to accelerate to the max speed, coast for a while, then apply
negative max input to decelerate to a stop at the desired position.
Optimization problems can get more complex than this though. In fact,
we can use this same framework to design optimal trajectories for a
drivetrain while satisfying dynamics constraints, avoiding obstacles,
and driving through points of interest.

## Optimizing the problem formulation

Cost functions and constraints can have the following orders:

* none (i.e., there is no cost function or are no constraints)

* constant

* linear

* quadratic

* nonlinear

For nonlinear problems, the solver calculates the Hessian of the cost
function and the Jacobians of the constraints at each iteration.
However, problems with lower order cost functions and constraints can
be solved faster. For example, the following only need to be computed
once because they're constant:

* the Hessian of a quadratic or lower cost function

* the Jacobian of linear or lower constraints

A problem is constant if:

* the cost function is constant or lower

* the equality constraints are constant or lower

* the inequality constraints are constant or lower

A problem is linear if:

* the cost function is linear

* the equality constraints are linear or lower

* the inequality constraints are linear or lower

A problem is quadratic if:

* the cost function is quadratic

* the equality constraints are linear or lower

* the inequality constraints are linear or lower

All other problems are nonlinear.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Callback =
R"doc(Sets a callback to be called at each solver iteration.

The callback for this overload should return void.

Parameter ``callback``:
    The callback.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Callback_2 =
R"doc(Sets a callback to be called at each solver iteration.

The callback for this overload should return bool.

Parameter ``callback``:
    The callback. Returning true from the callback causes the solver
    to exit early with the solution it has so far.)doc";

static const char *__doc_sleipnir_OptimizationProblem_DecisionVariable = R"doc(Create a decision variable in the optimization problem.)doc";

static const char *__doc_sleipnir_OptimizationProblem_DecisionVariable_2 =
R"doc(Create a matrix of decision variables in the optimization problem.

Parameter ``rows``:
    Number of matrix rows.

Parameter ``cols``:
    Number of matrix columns.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Maximize =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Maximize_2 =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``objective``:
    The objective function to maximize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Minimize =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Minimize_2 =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Parameter ``cost``:
    The cost function to minimize.)doc";

static const char *__doc_sleipnir_OptimizationProblem_OptimizationProblem = R"doc(Construct the optimization problem.)doc";

static const char *__doc_sleipnir_OptimizationProblem_Solve =
R"doc(Solve the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Parameter ``config``:
    Configuration options for the solver.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_2 =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_3 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SubjectTo_4 =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Parameter ``constraint``:
    The constraint to satisfy.)doc";

static const char *__doc_sleipnir_OptimizationProblem_SymmetricDecisionVariable =
R"doc(Create a symmetric matrix of decision variables in the optimization
problem.

Variable instances are reused across the diagonal, which helps reduce
problem dimensionality.

Parameter ``rows``:
    Number of matrix rows.)doc";

static const char *__doc_sleipnir_OptimizationProblem_m_callback = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_decisionVariables = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_equalityConstraints = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_f = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_m_inequalityConstraints = R"doc()doc";

static const char *__doc_sleipnir_OptimizationProblem_status = R"doc()doc";

static const char *__doc_sleipnir_Profiler =
R"doc(Records the number of profiler measurements (start/stop pairs) and the
average duration between each start and stop call.)doc";

static const char *__doc_sleipnir_Profiler_AverageSolveDuration = R"doc(The average solve duration in milliseconds as a double.)doc";

static const char *__doc_sleipnir_Profiler_SetupDuration = R"doc(The setup duration in milliseconds as a double.)doc";

static const char *__doc_sleipnir_Profiler_SolveMeasurements = R"doc(The number of solve measurements taken.)doc";

static const char *__doc_sleipnir_Profiler_StartSetup = R"doc(Tell the profiler to start measuring setup time.)doc";

static const char *__doc_sleipnir_Profiler_StartSolve = R"doc(Tell the profiler to start measuring solve time.)doc";

static const char *__doc_sleipnir_Profiler_StopSetup = R"doc(Tell the profiler to stop measuring setup time.)doc";

static const char *__doc_sleipnir_Profiler_StopSolve =
R"doc(Tell the profiler to stop measuring solve time, increment the number
of averages, and incorporate the latest measurement into the average.)doc";

static const char *__doc_sleipnir_Profiler_m_averageSolveDuration = R"doc()doc";

static const char *__doc_sleipnir_Profiler_m_setupDuration = R"doc()doc";

static const char *__doc_sleipnir_Profiler_m_setupStartTime = R"doc()doc";

static const char *__doc_sleipnir_Profiler_m_solveMeasurements = R"doc()doc";

static const char *__doc_sleipnir_Profiler_m_solveStartTime = R"doc()doc";

static const char *__doc_sleipnir_RK4 =
R"doc(Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for
dt.

Parameter ``f``:
    The function to integrate. It must take two arguments x and u.

Parameter ``x``:
    The initial value of x.

Parameter ``u``:
    The value u held constant over the integration period.

Parameter ``t0``:
    The initial time.

Parameter ``dt``:
    The time over which to integrate.)doc";

static const char *__doc_sleipnir_SolverConfig = R"doc(Solver configuration.)doc";

static const char *__doc_sleipnir_SolverConfig_acceptableTolerance =
R"doc(The solver will stop once the error is below this tolerance for
`acceptableIterations` iterations. This is useful in cases where the
solver might not be able to achieve the desired level of accuracy due
to floating-point round-off.)doc";

static const char *__doc_sleipnir_SolverConfig_diagnostics = R"doc(Enables diagnostic prints.)doc";

static const char *__doc_sleipnir_SolverConfig_feasibleIPM =
R"doc(Enables the feasible interior-point method. When the inequality
constraints are all feasible, step sizes are reduced when necessary to
prevent them becoming infeasible again. This is useful when parts of
the problem are ill-conditioned in infeasible regions (e.g., square
root of a negative value). This can slow or prevent progress toward a
solution though, so only enable it if necessary.)doc";

static const char *__doc_sleipnir_SolverConfig_maxAcceptableIterations =
R"doc(The solver will stop once the error is below `acceptableTolerance` for
this many iterations.)doc";

static const char *__doc_sleipnir_SolverConfig_maxIterations = R"doc(The maximum number of solver iterations before returning a solution.)doc";

static const char *__doc_sleipnir_SolverConfig_spy =
R"doc(Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named
H.spy, A_e.spy, and A_i.spy respectively during solve.

Use tools/spy.py to plot them.)doc";

static const char *__doc_sleipnir_SolverConfig_timeout = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

static const char *__doc_sleipnir_SolverConfig_tolerance = R"doc(The solver will stop once the error is below this tolerance.)doc";

static const char *__doc_sleipnir_SolverExitCondition = R"doc(Solver exit condition.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kCallbackRequestedStop =
R"doc(The solver returned its solution so far after the user requested a
stop.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kDivergingIterates =
R"doc(The solver encountered diverging primal iterates xₖ and/or sₖ and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kFeasibilityRestorationFailed =
R"doc(The solver failed to reach the desired tolerance, and feasibility
restoration failed to converge.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kLocallyInfeasible =
R"doc(The solver determined the problem to be locally infeasible and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kMaxIterationsExceeded =
R"doc(The solver returned its solution so far after exceeding the maximum
number of iterations.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kMaxWallClockTimeExceeded =
R"doc(The solver returned its solution so far after exceeding the maximum
elapsed wall clock time.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kNonfiniteInitialCostOrConstraints =
R"doc(The solver encountered nonfinite initial cost or constraints and gave
up.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kSolvedToAcceptableTolerance =
R"doc(Solved the problem to an acceptable tolerance, but not the desired
one.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kSuccess = R"doc(Solved the problem to the desired tolerance.)doc";

static const char *__doc_sleipnir_SolverExitCondition_kTooFewDOFs = R"doc(The solver determined the problem to be overconstrained and gave up.)doc";

static const char *__doc_sleipnir_SolverIterationInfo = R"doc(Solver iteration information exposed to a user callback.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_A_e = R"doc(The equality constraint Jacobian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_A_i = R"doc(The inequality constraint Jacobian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_H = R"doc(The Hessian of the Lagrangian.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_g = R"doc(The gradient of the cost function.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_iteration = R"doc(The solver iteration.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_s = R"doc(The inequality constraint slack variables.)doc";

static const char *__doc_sleipnir_SolverIterationInfo_x = R"doc(The decision variables.)doc";

static const char *__doc_sleipnir_SolverStatus =
R"doc(Return value of OptimizationProblem::Solve() containing the cost
function and constraint types and solver's exit condition.)doc";

static const char *__doc_sleipnir_SolverStatus_costFunctionType = R"doc(The cost function type detected by the solver.)doc";

static const char *__doc_sleipnir_SolverStatus_equalityConstraintType = R"doc(The equality constraint type detected by the solver.)doc";

static const char *__doc_sleipnir_SolverStatus_exitCondition = R"doc(The solver's exit condition.)doc";

static const char *__doc_sleipnir_SolverStatus_inequalityConstraintType = R"doc(The inequality constraint type detected by the solver.)doc";

static const char *__doc_sleipnir_TimestepMethod = R"doc(Enum describing the type of system timestep.)doc";

static const char *__doc_sleipnir_TimestepMethod_kFixed = R"doc(The timestep is a fixed constant.)doc";

static const char *__doc_sleipnir_TimestepMethod_kVariable = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

static const char *__doc_sleipnir_TimestepMethod_kVariableSingle =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

static const char *__doc_sleipnir_ToMessage =
R"doc(Returns user-readable message corresponding to the exit condition.

Parameter ``exitCondition``:
    Solver exit condition.)doc";

static const char *__doc_sleipnir_TranscriptionMethod = R"doc(Enum describing an OCP transcription method.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kDirectCollocation =
R"doc(The trajectory is modeled as a series of cubic polynomials where the
centerpoint slope is constrained.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kDirectTranscription =
R"doc(Each state is a decision variable constrained to the integrated
dynamics of the previous state.)doc";

static const char *__doc_sleipnir_TranscriptionMethod_kSingleShooting =
R"doc(States depend explicitly as a function of all previous states and all
previous inputs.)doc";

static const char *__doc_sleipnir_Variable = R"doc(An autodiff variable pointing to an expression node.)doc";

static const char *__doc_sleipnir_VariableBlock =
R"doc(A submatrix of autodiff variables with reference semantics.

Template parameter ``Mat``:
    The type of the matrix whose storage this class points to.)doc";

static const char *__doc_sleipnir_VariableBlock_Block =
R"doc(Returns a block slice of the variable matrix.

Parameter ``rowOffset``:
    The row offset of the block selection.

Parameter ``colOffset``:
    The column offset of the block selection.

Parameter ``blockRows``:
    The number of rows in the block selection.

Parameter ``blockCols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableBlock_Block_2 =
R"doc(Returns a block slice of the variable matrix.

Parameter ``rowOffset``:
    The row offset of the block selection.

Parameter ``colOffset``:
    The column offset of the block selection.

Parameter ``blockRows``:
    The number of rows in the block selection.

Parameter ``blockCols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableBlock_Col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Cols = R"doc(Returns number of columns in the matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_CwiseTransform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unaryOp``:
    The unary operator to use for the transform operation.)doc";

static const char *__doc_sleipnir_VariableBlock_Row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableBlock_Rows = R"doc(Returns number of rows in the matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_SetValue =
R"doc(Assigns a double to the block.

This only works for blocks with one row and one column.)doc";

static const char *__doc_sleipnir_VariableBlock_SetValue_2 = R"doc(Sets block's internal values.)doc";

static const char *__doc_sleipnir_VariableBlock_T = R"doc(Returns the transpose of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_Value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.)doc";

static const char *__doc_sleipnir_VariableBlock_Value_2 =
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.)doc";

static const char *__doc_sleipnir_VariableBlock_Value_3 = R"doc(Returns the contents of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_3 =
R"doc(Constructs a Variable block pointing to all of the given matrix.

Parameter ``mat``:
    The matrix to which to point.)doc";

static const char *__doc_sleipnir_VariableBlock_VariableBlock_4 =
R"doc(Constructs a Variable block pointing to a subset of the given matrix.

Parameter ``mat``:
    The matrix to which to point.

Parameter ``rowOffset``:
    The block's row offset.

Parameter ``colOffset``:
    The block's column offset.

Parameter ``blockRows``:
    The number of rows in the block.

Parameter ``blockCols``:
    The number of columns in the block.)doc";

static const char *__doc_sleipnir_VariableBlock_begin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_begin_2 = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_cbegin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_cend = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_m_col = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_m_row = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_end = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_end_2 = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableBlock_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_m_col = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_m_row = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_blockCols = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_blockRows = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_colOffset = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_m_rowOffset = R"doc()doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign = R"doc(Assigns a VariableBlock to the block.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_2 = R"doc(Assigns a VariableBlock to the block.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_3 =
R"doc(Assigns a double to the block.

This only works for blocks with one row and one column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_4 = R"doc(Assigns an Eigen matrix to the block.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_5 = R"doc(Assigns a VariableMatrix to the block.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_assign_6 = R"doc(Assigns a VariableMatrix to the block.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_call =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_call_2 =
R"doc(Returns a scalar subblock at the given row and column.

Parameter ``row``:
    The scalar subblock's row.

Parameter ``col``:
    The scalar subblock's column.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_call_3 =
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_call_4 =
R"doc(Returns a scalar subblock at the given row.

Parameter ``row``:
    The scalar subblock's row.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_iadd =
R"doc(Compound addition-assignment operator.

Parameter ``rhs``:
    Variable to add.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_idiv =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_idiv_2 =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_imul =
R"doc(Compound matrix multiplication-assignment operator.

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_imul_2 =
R"doc(Compound matrix multiplication-assignment operator (only enabled when
lhs is a scalar).

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableBlock_operator_isub =
R"doc(Compound subtraction-assignment operator.

Parameter ``rhs``:
    Variable to subtract.)doc";

static const char *__doc_sleipnir_VariableBlock_size = R"doc(Returns number of elements in matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix = R"doc(A matrix of autodiff variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_Block =
R"doc(Returns a block slice of the variable matrix.

Parameter ``rowOffset``:
    The row offset of the block selection.

Parameter ``colOffset``:
    The column offset of the block selection.

Parameter ``blockRows``:
    The number of rows in the block selection.

Parameter ``blockCols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableMatrix_Block_2 =
R"doc(Returns a block slice of the variable matrix.

Parameter ``rowOffset``:
    The row offset of the block selection.

Parameter ``colOffset``:
    The column offset of the block selection.

Parameter ``blockRows``:
    The number of rows in the block selection.

Parameter ``blockCols``:
    The number of columns in the block selection.)doc";

static const char *__doc_sleipnir_VariableMatrix_Col =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Col_2 =
R"doc(Returns a column slice of the variable matrix.

Parameter ``col``:
    The column to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Cols = R"doc(Returns number of columns in the matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_CwiseTransform =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Parameter ``unaryOp``:
    The unary operator to use for the transform operation.)doc";

static const char *__doc_sleipnir_VariableMatrix_Ones =
R"doc(Returns a variable matrix filled with ones.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_Row =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Row_2 =
R"doc(Returns a row slice of the variable matrix.

Parameter ``row``:
    The row to slice.)doc";

static const char *__doc_sleipnir_VariableMatrix_Rows = R"doc(Returns number of rows in the matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_Segment =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableMatrix_Segment_2 =
R"doc(Returns a segment of the variable vector.

Parameter ``offset``:
    The offset of the segment.

Parameter ``length``:
    The length of the segment.)doc";

static const char *__doc_sleipnir_VariableMatrix_SetValue = R"doc(Sets the VariableMatrix's internal values.)doc";

static const char *__doc_sleipnir_VariableMatrix_T = R"doc(Returns the transpose of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value =
R"doc(Returns an element of the variable matrix.

Parameter ``row``:
    The row of the element to return.

Parameter ``col``:
    The column of the element to return.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value_2 =
R"doc(Returns a row of the variable column vector.

Parameter ``index``:
    The index of the element to return.)doc";

static const char *__doc_sleipnir_VariableMatrix_Value_3 = R"doc(Returns the contents of the variable matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix = R"doc(Constructs an empty VariableMatrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_2 =
R"doc(Constructs a VariableMatrix column vector with the given rows.

Parameter ``rows``:
    The number of matrix rows.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_3 =
R"doc(Constructs a VariableMatrix with the given dimensions.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_4 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_5 =
R"doc(Constructs a scalar VariableMatrix from a nested list of doubles.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_6 =
R"doc(Constructs a scalar VariableMatrix from a nested list of Variables.

This overload is for Python bindings only.

Parameter ``list``:
    The nested list of Variables.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_7 = R"doc(Constructs a VariableMatrix from an Eigen matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_8 = R"doc(Constructs a VariableMatrix from an Eigen diagonal matrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_9 = R"doc(Constructs a scalar VariableMatrix from a Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_10 = R"doc(Constructs a scalar VariableMatrix from a Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_11 = R"doc(Constructs a VariableMatrix from a VariableBlock.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_12 = R"doc(Constructs a VariableMatrix from a VariableBlock.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_13 =
R"doc(Constructs a column vector wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.)doc";

static const char *__doc_sleipnir_VariableMatrix_VariableMatrix_14 =
R"doc(Constructs a matrix wrapper around a Variable array.

Parameter ``values``:
    Variable array to wrap.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_Zero =
R"doc(Returns a variable matrix filled with zeroes.

Parameter ``rows``:
    The number of matrix rows.

Parameter ``cols``:
    The number of matrix columns.)doc";

static const char *__doc_sleipnir_VariableMatrix_begin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_begin_2 = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_cbegin = R"doc(Returns begin iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_cend = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_const_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_m_col = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_m_row = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_const_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_end = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_end_2 = R"doc(Returns end iterator.)doc";

static const char *__doc_sleipnir_VariableMatrix_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_iterator = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_m_col = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_m_mat = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_m_row = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_eq = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_inc = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_iterator_operator_mul = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_cols = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_rows = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_m_storage = R"doc()doc";

static const char *__doc_sleipnir_VariableMatrix_operator_Variable = R"doc(Implicit conversion operator from 1x1 VariableMatrix to Variable.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_assign = R"doc(Assigns an Eigen matrix to a VariableMatrix.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_call =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_call_2 =
R"doc(Returns a block pointing to the given row and column.

Parameter ``row``:
    The block row.

Parameter ``col``:
    The block column.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_call_3 =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_call_4 =
R"doc(Returns a block pointing to the given row.

Parameter ``row``:
    The block row.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_iadd =
R"doc(Compound addition-assignment operator.

Parameter ``rhs``:
    Variable to add.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_idiv =
R"doc(Compound matrix division-assignment operator (only enabled when rhs is
a scalar).

Parameter ``rhs``:
    Variable to divide.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_imul =
R"doc(Compound matrix multiplication-assignment operator.

Parameter ``rhs``:
    Variable to multiply.)doc";

static const char *__doc_sleipnir_VariableMatrix_operator_isub =
R"doc(Compound subtraction-assignment operator.

Parameter ``rhs``:
    Variable to subtract.)doc";

static const char *__doc_sleipnir_VariableMatrix_size = R"doc(Returns number of elements in matrix.)doc";

static const char *__doc_sleipnir_Variable_SetValue =
R"doc(Sets Variable's internal value.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_SetValue_2 =
R"doc(Sets Variable's internal value.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_Type =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).)doc";

static const char *__doc_sleipnir_Variable_Update =
R"doc(Updates the value of this variable based on the values of its
dependent variables.)doc";

static const char *__doc_sleipnir_Variable_Value = R"doc(Returns the value of this variable.)doc";

static const char *__doc_sleipnir_Variable_Variable = R"doc(Constructs a linear Variable with a value of zero.)doc";

static const char *__doc_sleipnir_Variable_Variable_2 =
R"doc(Constructs a Variable from a double.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_Variable_3 =
R"doc(Constructs a Variable from an int.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_Variable_4 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_sleipnir_Variable_Variable_5 =
R"doc(Constructs a Variable pointing to the specified expression.

Parameter ``expr``:
    The autodiff variable.)doc";

static const char *__doc_sleipnir_Variable_expr = R"doc(The expression node.)doc";

static const char *__doc_sleipnir_Variable_operator_assign =
R"doc(Assignment operator for double.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_operator_assign_2 =
R"doc(Assignment operator for int.

Parameter ``value``:
    The value of the Variable.)doc";

static const char *__doc_sleipnir_Variable_operator_iadd =
R"doc(Variable-Variable compound addition operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_idiv =
R"doc(Variable-Variable compound division operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_imul =
R"doc(Variable-Variable compound multiplication operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_Variable_operator_isub =
R"doc(Variable-Variable compound subtraction operator.

Parameter ``rhs``:
    Operator right-hand side.)doc";

static const char *__doc_sleipnir_abs =
R"doc(std::abs() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_acos =
R"doc(std::acos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_asin =
R"doc(std::asin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_atan =
R"doc(std::atan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_atan2 =
R"doc(std::atan2() for Variables.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_sleipnir_cos =
R"doc(std::cos() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_cosh =
R"doc(std::cosh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_Expression = R"doc(An autodiff expression node.)doc";

static const char *__doc_sleipnir_detail_Expression_2 = R"doc(An autodiff expression node.)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph =
R"doc(This class is an adaptor type that performs value updates of an
expression's computational graph in a way that skips duplicates.)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_2 = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_ComputeAdjoints =
R"doc(Updates the adjoints in the expression graph, effectively computing
the gradient.

Parameter ``func``:
    A function that takes two arguments: an int for the gradient row,
    and a double for the adjoint (gradient value).)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_ExpressionGraph =
R"doc(Generates the deduplicated computational graph for the given
expression.

Parameter ``root``:
    The root node of the expression.)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_GenerateGradientTree =
R"doc(Returns the variable's gradient tree.

Parameter ``wrt``:
    Variables with respect to which to compute the gradient.)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_Update =
R"doc(Update the values of all nodes in this computational tree based on the
values of their dependent nodes.)doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_m_adjointList = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_m_rowList = R"doc()doc";

static const char *__doc_sleipnir_detail_ExpressionGraph_m_valueList = R"doc()doc";

static const char *__doc_sleipnir_detail_Expression_Expression = R"doc(Constructs a constant expression with a value of zero.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_2 =
R"doc(Constructs a nullary expression (an operator with no arguments).

Parameter ``value``:
    The expression value.

Parameter ``type``:
    The expression type. It should be either constant (the default) or
    linear.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_3 =
R"doc(Constructs an unary expression (an operator with one argument).

Parameter ``type``:
    The expression's type.

Parameter ``valueFunc``:
    Unary operator that produces this expression's value.

Parameter ``lhsGradientValueFunc``:
    Gradient with respect to the operand.

Parameter ``lhsGradientFunc``:
    Gradient with respect to the operand.

Parameter ``lhs``:
    Unary operator's operand.)doc";

static const char *__doc_sleipnir_detail_Expression_Expression_4 =
R"doc(Constructs a binary expression (an operator with two arguments).

Parameter ``type``:
    The expression's type.

Parameter ``valueFunc``:
    Unary operator that produces this expression's value.

Parameter ``lhsGradientValueFunc``:
    Gradient with respect to the left operand.

Parameter ``rhsGradientValueFunc``:
    Gradient with respect to the right operand.

Parameter ``lhsGradientFunc``:
    Gradient with respect to the left operand.

Parameter ``rhsGradientFunc``:
    Gradient with respect to the right operand.

Parameter ``lhs``:
    Binary operator's left operand.

Parameter ``rhs``:
    Binary operator's right operand.)doc";

static const char *__doc_sleipnir_detail_Expression_IsConstant =
R"doc(Returns true if the expression is the given constant.

Parameter ``constant``:
    The constant.)doc";

static const char *__doc_sleipnir_detail_Expression_adjoint = R"doc(The adjoint of the expression node used during autodiff.)doc";

static const char *__doc_sleipnir_detail_Expression_adjointExpr =
R"doc(The adjoint of the expression node used during gradient expression
tree generation.)doc";

static const char *__doc_sleipnir_detail_Expression_args = R"doc(Expression arguments.)doc";

static const char *__doc_sleipnir_detail_Expression_duplications =
R"doc(Tracks the number of instances of this expression yet to be
encountered in an expression tree.)doc";

static const char *__doc_sleipnir_detail_Expression_gradientFuncs =
R"doc(Functions returning Variable adjoints of the children expressions.

Parameters:

* lhs: Left argument to binary operator.

* rhs: Right argument to binary operator.

* parentAdjoint: Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_gradientValueFuncs =
R"doc(Functions returning double adjoints of the children expressions.

Parameters:

* lhs: Left argument to binary operator.

* rhs: Right argument to binary operator.

* parentAdjoint: Adjoint of parent expression.)doc";

static const char *__doc_sleipnir_detail_Expression_refCount = R"doc(Reference count for intrusive shared pointer.)doc";

static const char *__doc_sleipnir_detail_Expression_row =
R"doc(This expression's row in wrt for autodiff gradient, Jacobian, or
Hessian. This is -1 if the expression isn't in wrt.)doc";

static const char *__doc_sleipnir_detail_Expression_type = R"doc(Expression argument type.)doc";

static const char *__doc_sleipnir_detail_Expression_value = R"doc(The value of the expression node.)doc";

static const char *__doc_sleipnir_detail_Expression_valueFunc =
R"doc(Either nullary operator with no arguments, unary operator with one
argument, or binary operator with two arguments. This operator is used
to update the node's value.)doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrDecRefCount =
R"doc(Refcount decrement for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_IntrusiveSharedPtrIncRefCount =
R"doc(Refcount increment for intrusive shared pointer.

Parameter ``expr``:
    The shared pointer's managed object.)doc";

static const char *__doc_sleipnir_detail_MakeExpressionPtr =
R"doc(Creates an intrusive shared pointer to an expression from the global
pool allocator.

Parameter ``args``:
    Constructor arguments for Expression.)doc";

static const char *__doc_sleipnir_detail_Zero =
R"doc(Returns an instance of "zero", which has special meaning in expression
operations.)doc";

static const char *__doc_sleipnir_detail_abs =
R"doc(std::abs() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_acos =
R"doc(std::acos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_asin =
R"doc(std::asin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_atan =
R"doc(std::atan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_atan2 =
R"doc(std::atan2() for Expressions.

Parameter ``y``:
    The y argument.

Parameter ``x``:
    The x argument.)doc";

static const char *__doc_sleipnir_detail_cos =
R"doc(std::cos() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_cosh =
R"doc(std::cosh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_erf =
R"doc(std::erf() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_exp =
R"doc(std::exp() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_hypot =
R"doc(std::hypot() for Expressions.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_sleipnir_detail_log =
R"doc(std::log() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_log10 =
R"doc(std::log10() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_pow =
R"doc(std::pow() for Expressions.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_sleipnir_detail_sign =
R"doc(sign() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sin =
R"doc(std::sin() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sinh =
R"doc(std::sinh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_sqrt =
R"doc(std::sqrt() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_tan =
R"doc(std::tan() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_detail_tanh =
R"doc(std::tanh() for Expressions.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_erf =
R"doc(std::erf() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_exp =
R"doc(std::exp() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_hypot =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.)doc";

static const char *__doc_sleipnir_hypot_2 =
R"doc(std::hypot() for Variables.

Parameter ``x``:
    The x argument.

Parameter ``y``:
    The y argument.

Parameter ``z``:
    The z argument.)doc";

static const char *__doc_sleipnir_log =
R"doc(std::log() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_log10 =
R"doc(std::log10() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_operator_eq =
R"doc(Equality operator that returns an equality constraint for two
Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_ge =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_gt =
R"doc(Greater-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_le =
R"doc(Less-than-or-equal-to comparison operator that returns an inequality
constraint for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_operator_lshift =
R"doc(Catch2 value formatter for ExpressionType.

Parameter ``os``:
    Output stream to which to print.

Parameter ``type``:
    ExpressionType to print.)doc";

static const char *__doc_sleipnir_operator_lshift_2 =
R"doc(Catch2 value formatter for SolverExitCondition.

Parameter ``os``:
    Output stream to which to print.

Parameter ``exitCondition``:
    Solver exit condition to print.)doc";

static const char *__doc_sleipnir_operator_lt =
R"doc(Less-than comparison operator that returns an inequality constraint
for two Variables.

Parameter ``lhs``:
    Left-hand side.

Parameter ``rhs``:
    Left-hand side.)doc";

static const char *__doc_sleipnir_pow =
R"doc(std::pow() for Variables.

Parameter ``base``:
    The base.

Parameter ``power``:
    The power.)doc";

static const char *__doc_sleipnir_sign =
R"doc(sign() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_sin =
R"doc(std::sin() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_sinh =
R"doc(std::sinh() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_sqrt =
R"doc(std::sqrt() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_tan =
R"doc(std::tan() for Variables.

Parameter ``x``:
    The argument.)doc";

static const char *__doc_sleipnir_tanh =
R"doc(std::tanh() for Variables.

Parameter ``x``:
    The argument.)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

