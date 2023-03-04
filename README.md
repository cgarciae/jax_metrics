<!-- codecov badge -->
[![codecov](https://codecov.io/gh/cgarciae/jax_metrics/branch/master/graph/badge.svg?token=3IKEUAU3C8)](https://codecov.io/gh/cgarciae/jax_metrics)

# JAX Metrics

_A Metrics library for the JAX ecosystem_

#### Main Features
* Standard metrics that can be used in any JAX project.
* Pytree abstractions that can natively integrate with all JAX APIs and pytree-supporting frameworks (flax.struct, equinox, treex, etc).
* Distributed-friendly APIs that make it super easy to synchronize metrics across devices.
* Automatic accumulation over epochs.


JAX Metrics is implemented on top of [Treeo](https://github.com/cgarciae/treeo).

## What is included?
* The Keras-like `Loss` and `Metric` abstractions.
* A `metrics` module containing popular metrics.
* The `losses` and `regularizers` modules containing popular losses.
* The `Metrics` and `Losses` combinators.

<!-- ## Why JAX Metrics? -->

## Installation
Install using pip:
```bash
pip install jax_metrics
```

## Status
Metrics on this library are usually tested against their Keras or Torchmetrics counterparts for numerical equivalence. This code base comes from Treex and Elegy so it's already in use.

## Getting Started

### Metric

The `Metric` API consists of 3 basic methods:

* `reset`: Used to both initialize and reset a metric.
* `update`: Takes in new data and updates the metric state.
* `compute`: Returns the current value of the metric.

Simple usage looks like this:


```python
import jax_metrics as jm

metric = jm.metrics.Accuracy()

# Update the metric with a batch of predictions and labels
metric = metric.update(target=y, preds=logits)

# Get the current value of the metric
acc = metric.compute() # 0.95

# alternatively, produce a logs dict
logs = metric.compute_logs() # {'accuracy': 0.95}

# Reset the metric
metric = metric.reset()
```

Note that `update` enforces the use of keyword arguments. Also the `Metric.name` property is used as the key in the returned dict, by default this is the name of the class in lowercase but can be overridden in the constructor via the `name` argument.

#### Tipical Training Setup

Because Metrics are pytrees they can be used with `jit`, `pmap`, etc. On a more realistic scenario you will proably want to use them inside some of your JAX functions in a setup similar to this:

```python
import jax_metrics as jm

metric = jm.metrics.Accuracy()

def loss_fn(params, metric, x, y):
    ...
    metric = metric.update(target=y, preds=logits)
    ...

    return loss, metric

@jax.jit
def train_step(params, metric, x, y):
    grads, metric = jax.grad(loss_fn, has_aux=True)(
        params, metric, x, y
    )
    ...
    return params, metric

@jax.jit
def reset_step(metric: jm.Metric) -> jm.Metric:
    return metric.reset()
```

Since the loss function usually has access to the predictions and labels, its usually where you would call `metric.update`, and the new metric state can be returned as an auxiliary output.

#### Distributed Training

JAX Metrics has a distributed friendly API via the `batch_updates` and `reduce` methods. A simple example of a loss function inside a data parallel setup could look like this:

```python
def loss_fn(params, metric, x, y):
    ...
    # compuate batch update
    batch_updates = metric.batch_updates(target=y, preds=logits)
    # gather over all devices and reduce
    batch_updates = jax.lax.all_gather(batch_updates, "device").reduce()
    # update metric
    metric = metric.merge(batch_updates)
    ...
```

The `batch_updates` method behaves similar to `update` but returns a new metric state with only information about that batch, `jax.lax.all_gather` "gathers" the metric state over all devices plus adds a new axis to the metric state, and `reduce` reduces the metric state over all devices (first axis). Finally, `merge` combines the accumulated metric state over the previous batches with the batch updates.

### Loss

The `Loss` API just consists of a `__call__` method. Simple usage looks like this:

```python
import jax_metrics as jm

crossentropy = jm.losses.Crossentropy()

# get reduced loss value
loss = crossentropy(target=y, preds=logits) # 0.23
```
Note that losses are not pytrees so they should be marked as static. Similar to Keras, all losses have a `reduction` strategy that can be specified in the constructor and (usually) makes sure that the output is a scalar.

<details>
<summary><b>Why have losses in a metrics library?</b></summary>
<!-- #### Why have losses in a metrics library? -->

There are a few reasons for having losses in a metrics library:

1. Most code from this library was originally written for and will still be consumed by Elegy. Since Elegy needs support for calculating cumulative losses, as you will see later, a Metric abstraction called `Losses` was created for this.
2. A couple of API design decisions are shared between the `Loss` and `Metric` APIs. This includes: 
    * `__call__` and `update` both accept any number keyword only arguments. This is used to facilitate composition (see [Combinators](#combinators) section).
    * Both classes have the `index_into` and `rename_arguments` methods that allow them to modify how arguments are consumed.
    * Argument names are standardized to be consistent when ever possible, e.g. both `metrics.Accuracy` and `losses.Crossentropy` use the `target` and `preds` arguments.

</details>

### Combinators
Combinators enable you to group together multiple metrics while also being instances of `Metric` and thus maintaining the same API.
#### Metrics
The `Metrics` combinator lets you combine multiple metrics into a single Metric object.

```python
metrics = jm.Metrics([
    jm.metrics.Accuracy(),
    jm.metrics.F1(), # not yet implemented 😅, coming soon?
])

# same API
metrics = metrics.update(target=y, preds=logits)
# compute now returns a dict
metrics.compute() # {'accuracy': 0.95, 'f1': 0.87}
# same as compute_logs in the case
metrics.compute_logs() # {'accuracy': 0.95, 'f1': 0.87}
# Reset the metrics
metrics = metrics.reset()
```

As you can see the `Metrics.update` method accepts and forwards all the arguments required by the individual metrics. In this example they use the same arguments, but in practice they may consume different subsets of the arguments. Also, if names are repeated then unique names are generated for each metric by appending a number to the metric name.

If a dictionary is used instead of a list, the keys are used instead of the `name` property of the metrics to determine the key in the returned dict.

```python
metrics = jm.Metrics({
    "acc": jm.metrics.Accuracy(),
    "f_one": jm.metrics.F1(), # not yet implemented 😅, coming soon?
})

# same API
metrics = metrics.update(target=y, preds=logits)
# compute new returns a dict
metrics.compute() # {'acc': 0.95, 'f_one': 0.87}
# same as compute_logs in the case
metrics.compute_logs() # {'acc': 0.95, 'f_one': 0.87}
# Reset the metrics
metrics = metrics.reset()
```

You can use nested structures of dicts and lists to group metrics, the keys of the dicts are used to determine group names. Group names and metrics names are concatenated using `"/"` e.g. `"{group_name}/{metric_name}"`.

#### Losses

`Losses` is a `Metric` combinator that behaves very similarly to `Metrics` but contains `Loss` instances. `Losses` calculates the cumulative **mean** value of each loss over the batches.

```python
losses = jm.Losses([
    jm.losses.Crossentropy(),
    jm.regularizers.L2(1e-4),
])

# same API
losses = losses.update(target=y, preds=logits, parameters=params)
# compute new returns a dict
losses.compute() # {'crossentropy': 0.23, 'l2': 0.005}
# same as compute_logs in the case
losses.compute_logs() # {'crossentropy': 0.23, 'l2': 0.005}
# you can also compute the total loss
loss = losses.total_loss() # 0.235
# Reset the losses
losses = losses.reset()
```

As with `Metrics`, the `update` method accepts and forwards all the arguments required by the individual losses. In this example `target` and `preds` are used by the `Crossentropy`, while `parameters` is used by the `L2`. The `total_loss` method returns the sum of all values returned by `compute`.

If a dictionary is used instead of a list, the keys are used instead of the `name` property of the losses to determine the key in the returned dict.

```python
losses = jm.Losses({
    "xent": jm.losses.Crossentropy(),
    "l2": jm.regularizers.L2(1e-4),
})

# same API
losses = losses.update(target=y, preds=logits, parameters=params)
# compute new returns a dict
losses.compute() # {'xent': 0.23, 'l2': 0.005}
# same as compute_logs in the case
losses.compute_logs() # {'xent': 0.23, 'l2': 0.005}
# you can also compute the total loss
loss = losses.total_loss() # 0.235
# Reset the losses
losses = losses.reset()
```

If you want to use `Losses` to calculate the loss of a model, you should use `batch_updates` followed by `total_loss` to get the correct batch loss. For example, a loss function could be written as:

```python
def loss_fn(..., losses):
    ...
    batch_updates = losses.batch_updates(target=y, preds=logits, parameters=params)
    loss = batch_updates.total_loss()
    losses = losses.merge(batch_updates)
    ...
    return loss, losses
```
For convenience, the previous pattern can be simplified to a single line using the `loss_and_update` method:
```python
def loss_fn(...):
    ...
    loss, lossses = losses.loss_and_update(target=y, preds=logits, parameters=params)
    ...
    return loss, losses
```
