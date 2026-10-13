# Week 3: Momentum and learning-rate schedules

## Why plain gradient descent zig-zags

In a long narrow valley the gradient points mostly across the valley,
not along it. Plain gradient descent bounces between the walls and makes
slow progress toward the minimum. This is what an ill-conditioned
problem looks like: the curvature is very different in different
directions.

## Heavy-ball momentum

Momentum keeps a running velocity. Each step does
`v = beta * v + grad(loss)(theta)` and then `theta = theta - learning_rate * v`.
With beta around 0.9, steps that agree in direction add up and steps that
flip sign cancel out, so the zig-zag is damped.

## Nesterov momentum

Nesterov's variant evaluates the gradient at the look-ahead point
`theta - learning_rate * beta * v` instead of at theta. It corrects
course a step earlier and often converges a little faster on smooth
convex problems.

## Learning-rate schedules

A fixed learning rate is rarely best for the whole run. Step decay
divides the rate by a constant, such as 10, every few epochs. Cosine
annealing lowers it smoothly from the starting value to near zero along
half a cosine wave. A short warmup, where the rate grows linearly over
the first few hundred steps, avoids huge early updates.
