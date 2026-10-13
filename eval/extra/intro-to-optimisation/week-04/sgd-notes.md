# Week 4: Stochastic gradient descent and regularisation

## Minibatches

Computing the full gradient over every training example is expensive.
Stochastic gradient descent estimates it from a small random minibatch,
for example 32 or 128 examples. One pass over the whole training set is
an epoch.

## Noise is not all bad

The minibatch gradient is noisy. Larger batches reduce the noise but cost
more per step. A rough rule: when the batch size doubles, the learning
rate can often be raised as well, up to a point. Some noise can help the
iterate escape shallow regions of a non-convex loss.

## Weight decay

L2 regularisation adds `(alpha / 2) * ||theta||^2` to the loss. Its
gradient is alpha * theta, so every step also shrinks the weights toward
zero. That is why it is called weight decay.

## L1 and sparsity

L1 regularisation adds alpha times the sum of absolute weights. Because
its penalty does not flatten out near zero, it pushes many weights to
exactly zero, giving a sparse model.

## Early stopping

Keep a validation set aside and stop training when the validation loss
has not improved for a set number of epochs, called the patience. Keep
the weights from the best epoch.
