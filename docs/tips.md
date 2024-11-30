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

A problem is linear if:

* the cost function is linear
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

A problem is quadratic if:

* the cost function is quadratic
* the equality constraints are linear or lower
* the inequality constraints are linear or lower

All other problems are nonlinear.

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
