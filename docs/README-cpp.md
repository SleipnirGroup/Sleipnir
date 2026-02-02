# Tutorial

## Setup

See the
[C++ installation instructions](https://sleipnirgroup.github.io/Sleipnir/index.html#autotoc_md2).

## Introduction

A system with position and velocity states and an acceleration input is an
example of a double integrator. We want to go from 0 m at rest to 2 m at rest
in the minimum time while obeying the velocity limit (-1, 1) and the
acceleration limit (-1, 1).

The model for our double integrator is ẍ = u where x is the vector [position;
velocity] and u is the acceleration. The velocity constraints are -1 ≤ x₁ ≤ 1
and the acceleration constraints are -1 ≤ u ≤ 1.

## Importing required libraries

```cpp
#include <Eigen/Core>
#include <sleipnir/optimization/problem.hpp>
```

## Initializing a problem instance

First, we need to make a problem instance.

```cpp
constexpr auto TOTAL_TIME = 5s;
constexpr auto dt = 5ms;
constexpr int N = TOTAL_TIME / dt;

constexpr double r = 2.0;

slp::Problem problem;
```

## Creating decision variables

First, we need to make decision variables for our state and input.

```cpp
// 2x1 state vector with N + 1 timesteps (includes last state)
auto X = problem.decision_variable(2, N + 1);

// 1x1 input vector with N timesteps (input at last state doesn't matter)
auto U = problem.decision_variable(1, N);
```

By convention, we use capital letters for the variables to designate
matrices.

## Applying constraints

Now, we need to apply dynamics constraints between timesteps.

```cpp
// Kinematics constraint assuming constant acceleration between timesteps
for (int k = 0; k < N; ++k) {
  constexpr double t = std::chrono::duration<double>(dt).count();
  auto p_k1 = X(0, k + 1);
  auto v_k1 = X(1, k + 1);
  auto p_k = X(0, k);
  auto v_k = X(1, k);
  auto a_k = U(0, k);

  // pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
  problem.subject_to(p_k1 == p_k + v_k * t + 0.5 * a_k * t * t);

  // vₖ₊₁ = vₖ + aₖt
  problem.subject_to(v_k1 == v_k + a_k * t);
}
```

Next, we'll apply the state and input constraints.

```cpp
// Start and end at rest
problem.subject_to(X.Col(0) == Eigen::Matrix<double, 2, 1>{{0.0}, {0.0}});
problem.subject_to(X.Col(N) == Eigen::Matrix<double, 2, 1>{{r}, {0.0}});

// Limit velocity
problem.subject_to(-1 <= X.row(1));
problem.subject_to(X.row(1) <= 1);

// Limit acceleration
problem.subject_to(-1 <= U);
problem.subject_to(U <= 1);
```

## Specifying a cost function

Next, we'll create a cost function for minimizing position error.

```cpp
// Cost function - minimize position error
slp::Variable J = 0.0;
for (int k = 0; k < N + 1; ++k) {
  J += slp::pow(r - X(0, k), 2);
}
problem.minimize(J);
```

The cost function passed to minimize() should produce a scalar output.

## Solving the problem

Now we can solve the problem.

```cpp
problem.solve();
```

The solver will find the decision variable values that minimize the cost
function while satisfying the constraints.

## Accessing the solution

You can obtain the solution by querying the values of the variables like so.

```cpp
double position = X.value(0, 0);
double velocity = X.value(1, 0);
double acceleration = U.value(0);
```

## Other applications

In retrospect, the solution here seems obvious: if you want to reach the desired
position in the minimum time, you just apply positive max input to accelerate to
the max speed, coast for a while, then apply negative max input to decelerate to
a stop at the desired position. Optimization problems can get more complex than
this though. In fact, we can use this same framework to design optimal
trajectories for a drivetrain while satisfying dynamics constraints, avoiding
obstacles, and driving through points of interest.
