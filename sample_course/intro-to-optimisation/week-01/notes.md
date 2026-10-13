# Week 1: Gradient descent

My own notes for week 1.

## The update rule

Gradient descent moves the parameters a small step against the gradient:
`theta = theta - learning_rate * grad(loss)(theta)`. The learning rate
controls the step size.

## Choosing the learning rate

If the learning rate is too large, the loss oscillates or diverges. If it
is too small, training is slow. A common first check is to plot the loss
for a few learning rates spaced by a factor of ten.

```python
# A heading inside code is not a section:
# Not a heading
```

## Stopping

Stop when the gradient norm falls below a tolerance or after a fixed
number of iterations.
