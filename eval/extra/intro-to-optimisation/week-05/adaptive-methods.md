# Week 5: Adaptive learning-rate methods

## AdaGrad

AdaGrad keeps the running sum of squared gradients for each parameter and
divides the step by its square root. Parameters with rare, small
gradients get relatively larger steps. The sum only grows, so the
effective rate keeps shrinking, which can stop training too early.

## RMSProp

RMSProp replaces the sum with an exponential moving average,
`s = rho * s + (1 - rho) * g^2`, so old gradients are forgotten and the
step size does not decay to zero.

## Adam

Adam combines momentum with RMSProp. It keeps a first moment
`m = beta1 * m + (1 - beta1) * g` and a second moment
`v = beta2 * v + (1 - beta2) * g^2`. Because both start at zero, it
divides them by `1 - beta1^t` and `1 - beta2^t`; this is the bias
correction. The update is `theta = theta - lr * m_hat / (sqrt(v_hat) + eps)`.
Common defaults are beta1 = 0.9, beta2 = 0.999, eps = 1e-8.

## AdamW

Adding L2 regularisation to the loss interacts badly with Adam's
per-parameter scaling. AdamW instead applies weight decay directly to the
weights, separately from the gradient step.
