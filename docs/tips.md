# Tips

## Optimizing the problem formulation

Cost functions and constraints can have the following orders:

* none (i.e., there is no cost function or are no constraints)
* constant
* linear
* quadratic
* nonlinear

For nonlinear problems, the solver calculates the Hessian of the cost
function and the Jacobians of the constraints at each iteration. However,
problems with lower order cost functions and constraints can be solved
faster. For example, the following only need to be computed once because
they're constant:

* the Hessian of a quadratic or lower cost function
* the Jacobian of linear or lower constraints

A problem is constant if:

* the cost function is constant or lower
* the equality constraints are constant or lower
* the inequality constraints are constant or lower

A problem is a linear program (LP) if:

* the cost function is linear
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

A problem is a quadratic program (QP) if:

* the cost function is quadratic
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

All other problems are nonlinear programs (NLPs).

## Avoiding numerical issues

Instead of using distance (2-norm) for the cost function, use sum-of-squares.
The distance calculation's square root is nonlinear and has a limited domain,
whereas sum-of-squares has the same minimum, is quadratic, and has no domain
restriction. In other words, use `minimize(x ** 2 + y ** 2 + z ** 2)` instead of
`minimize(hypot(x, y, z))`.

## Deduplicating autodiff work

Store common subexpressions in intermediate variables and reuse them instead of
writing out the subexpressions each time. This ensures common subexpressions in
the expression tree are only traversed and updated once.

## Minimum-time problems

The obvious problem formulation for minimum-time problems uses one dt shared
across all timesteps.

```python
import sleipnir as slp

N = 100
T_max = 5.0

problem = slp.optimization.Problem()

x = problem.decision_variable(N + 1)
v = problem.decision_variable(N)

dt = problem.decision_variable()
dt.set_value(T_max / N)
problem.subject_to(dt > 0)
problem.subject_to(dt < T_max / N)

for k in range(N):
    problem.subject_to(x[k + 1] == x[k] + v[k] * dt)

problem.minimize(dt)

problem.solve()
```
The nonzero initial value for dt avoids a degenerate case, and the upper bound
prevents the solver exploiting discretization artifacts.

This formulation can have feasibility issues though per section 15.3
"Elimination of variables" of "Numerical Optimization, 2nd Ed.". Instead, we
recommend using a separate dt for each timestep, with them all
equality-constrained.

```python
import sleipnir as slp

N = 100
T_max = 5.0

problem = slp.optimization.Problem()

x = problem.decision_variable(N + 1)
v = problem.decision_variable(N)

dt = problem.decision_variable(N)
problem.subject_to(dt > 0)
problem.subject_to(dt < T_max / N)
for k in range(N - 1):
    problem.subject_to(dt[k] == dt[k + 1])

for k in range(N):
    dt[k].set_value(T_max / N)
    problem.subject_to(x[k + 1] == x[k] + v[k] * dt[k])

problem.minimize(sum(dt))

problem.solve()
```
