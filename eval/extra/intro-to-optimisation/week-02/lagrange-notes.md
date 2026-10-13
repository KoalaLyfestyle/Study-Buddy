# Week 2: Constrained optimisation

My notes on optimising with constraints. These follow the convexity lecture.

## Equality constraints

To minimise f(x) subject to g(x) = 0, form the Lagrangian
`L(x, lambda) = f(x) + lambda * g(x)` and look for points where its
gradient with respect to both x and lambda is zero. The number lambda is
called a Lagrange multiplier.

## Geometric picture

At a constrained optimum the gradient of f is parallel to the gradient of
g. Moving along the constraint surface cannot lower f to first order,
because the only direction that would lower it points off the surface.

## Inequality constraints and KKT

With constraints h(x) <= 0, the Karush-Kuhn-Tucker conditions add two
requirements: each multiplier mu must be non-negative, and complementary
slackness must hold, `mu * h(x) = 0`. So a multiplier can only be
non-zero when its constraint is active.

## Worked example

Minimise x^2 + y^2 subject to x + y = 1. Setting the gradient of
`x^2 + y^2 + lambda * (x + y - 1)` to zero gives 2x + lambda = 0 and
2y + lambda = 0, so x = y = 1/2 and lambda = -1. The minimum value is 1/2.
