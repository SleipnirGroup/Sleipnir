# Tutorial

## Setup

See the
[Python installation instructions](https://sleipnirgroup.github.io/Sleipnir/index.html#autotoc_md2).

## Introduction

A system with position and velocity states and an acceleration input is an
example of a double integrator. We want to go from 0 m at rest to 10 m at rest
in the minimum time while obeying the velocity limit (-1, 1) and the
acceleration limit (-1, 1).

The model for our double integrator is ẍ = u where x is the vector [position;
velocity] and u is the acceleration. The velocity constraints are -1 ≤ x(1) ≤ 1
and the acceleration constraints are -1 ≤ u ≤ 1.

## Importing required libraries

```py
from jormungandr.optimization import OptimizationProblem
import numpy as np
```

## Initializing a problem instance

First, we need to make a problem instance.

```py
T = 5.0  # s
dt = 0.005  # 5 ms
N = int(T / dt)

r = 2.0

problem = OptimizationProblem()
```

## Creating decision variables

First, we need to make decision variables for our state and input.

```py
# 2x1 state vector with N + 1 timesteps (includes last state)
X = problem.decision_variable(2, N + 1)

# 1x1 input vector with N timesteps (input at last state doesn't matter)
U = problem.decision_variable(1, N)
```

By convention, we use capital letters for the variables to designate
matrices.

## Applying constraints

Now, we need to apply dynamics constraints between timesteps.

```py
# Kinematics constraint assuming constant acceleration between timesteps
for k in range(N):
    p_k1 = X[0, k + 1]
    v_k1 = X[1, k + 1]
    p_k = X[0, k]
    v_k = X[1, k]
    a_k = U[0, k]

    # pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
    problem.subject_to(p_k1 == p_k + v_k * dt + 0.5 * a_k * dt**2)

    # vₖ₊₁ = vₖ + aₖt
    problem.subject_to(v_k1 == v_k + a_k * dt)
```

Next, we'll apply the state and input constraints.

```py
# Start and end at rest
problem.subject_to(X[:, 0] == np.array([[0.0], [0.0]]))
problem.subject_to(X[:, N] == np.array([[r], [0.0]]))

# Limit velocity
problem.subject_to(-1 <= X[1, :])
problem.subject_to(X[1, :] <= 1)

# Limit acceleration
problem.subject_to(-1 <= U)
problem.subject_to(U <= 1)
```

## Specifying a cost function

Next, we'll create a cost function for minimizing position error.

```py
# Cost function - minimize position error
J = 0.0
for k in range(N + 1):
    J += (r - X[0, k]) ** 2
problem.minimize(J)
```

The cost function passed to Minimize() should produce a scalar output.

## Solving the problem

Now we can solve the problem.

```py
problem.solve()
```

The solver will find the decision variable values that minimize the cost
function while satisfying the constraints.

## Accessing the solution

You can obtain the solution by querying the values of the variables like so.

```py
position = X.value(0, 0)
velocity = X.value(1, 0)
acceleration = U.value(0)
```

## Other applications

In retrospect, the solution here seems obvious: if you want to reach the desired
position in the minimum time, you just apply positive max input to accelerate to
the max speed, coast for a while, then apply negative max input to decelerate to
a stop at the desired position. Optimization problems can get more complex than
this though. In fact, we can use this same framework to design optimal
trajectories for a drivetrain while satisfying dynamics constraints, avoiding
obstacles, and driving through points of interest.
