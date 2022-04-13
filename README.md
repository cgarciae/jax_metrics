# JAX Metrics

_A Metrics library for the JAX ecosystem_

#### Main Features
* Standard framework-independent metrics that can be used in any JAX project.
* Pytree-based abstractions that can natively integrate with all JAX APIs.
* Distributed-friendly APIs that make it super easy to synchronize metrics across devices.
* Automatic accumulation over entire epochs.


JAX Metrics is implemented on top of [Treeo](https://github.com/cgarciae/treeo).

## What is included?
* A Keras-like `Metric` abstraction.
* A Keras-like `Loss` abstraction.
* A `Metrics`, `Losses`, and `LossesAndMetrics` combinators.
* A `metrics` moduel containing popular metrics.
* A `losses` and `regularizers` module containing popular losses.

<!-- ## Why JAX Metrics? -->

## Installation
Install using pip:
```bash
pip install jax_metrics
```

## Getting Started

```python
import jax_metrics as jm

metric = jm.metrics.Accuracy()

# Initialize the metric
metric = metric.reset()

# Update the metric with a batch of predictions and labels
metric = metric.update(target=y, preds=logits)

# Get the current value of the metric
acc = metric.compute() # 0.95

# alternatively, produce a logs dict
logs = metric.compute_logs() # {'accuracy': 0.95}
```

```python
import jax_metrics as jm

metric = jm.metrics.Accuracy()

@jax.jit
def init_step(metric: jm.Metric) -> jm.Metric:
    return metric.reset()


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
```

```python
def loss_fn(params, metric, x, y):
    ...
    # compuate batch update
    batch_updates = metric.batch_updates(target=y, preds=logits)
    # gather over all devices and aggregate
    batch_updates = jax.lax.all_gather(batch_updates, "device").aggregate()
    # update metric
    metric = metric.merge(batch_updates)
    ...
```

```python
batch_updates = jax.lax.psum(batch_updates, "device")
```

```python
metrics = jm.Metrics([
    jm.metrics.Accuracy(),
    jm.metrics.F1(), # not yet implemented 😅, coming soon?
])

# same API
metrics = metrics.reset()
# same API
metrics = metrics.update(target=y, preds=logits)
# compute new returns a dict
metrics.compute() # {'accuracy': 0.95, 'f1': 0.87}
# same as compute_logs in the case
metrics.compute_logs() # {'accuracy': 0.95, 'f1': 0.87}
```

```python
metrics = jm.Metrics({
    "acc": jm.metrics.Accuracy(),
    "f_one": jm.metrics.F1(), # not yet implemented 😅, coming soon?
})

# same API
metrics = metrics.reset()
# same API
metrics = metrics.update(target=y, preds=logits)
# compute new returns a dict
metrics.compute() # {'acc': 0.95, 'f_one': 0.87}
# same as compute_logs in the case
metrics.compute_logs() # {'acc': 0.95, 'f_one': 0.87}
```

```python
losses = jm.Losses([
    jm.losses.Crossentropy(),
    jm.regularizers.L2(1e-4),
])

# same API
losses = losses.reset()
# same API
losses = losses.update(target=y, preds=logits, parameters=params)
# compute new returns a dict
losses.compute() # {'crossentropy': 0.23, 'l2': 0.005}
# same as compute_logs in the case
losses.compute_logs() # {'crossentropy': 0.23, 'l2': 0.005}
# you can also compute the total loss
total_loss = losses.total_loss() # 0.235
```

```python
losses = jm.Losses({
    "xent": jm.losses.Crossentropy(),
    "l_two": jm.regularizers.L2(1e-4),
})

# same API
losses = losses.reset()
# same API
losses = losses.update(target=y, preds=logits, parameters=params)
# compute new returns a dict
losses.compute() # {'xent': 0.23, 'l_two': 0.005}
# same as compute_logs in the case
losses.compute_logs() # {'xent': 0.23, 'l_two': 0.005}
# you can also compute the total loss
total_loss = losses.total_loss() # 0.235
```

```python
def loss_fn(...):
    ...
    batch_updates = losses.loss_and_update(target=y, preds=logits, parameters=params)
    loss = batch_updates.total_loss()
    losses = losses.merge(batch_updates)
    ...
    return loss, losses
```

```python
def loss_fn(...):
    ...
    loss, lossses = losses.loss_and_update(target=y, preds=logits, parameters=params)
    ...
    return loss, losses
```

```python
lms = jm.LossesAndMetrics(
    metrics=[
        jm.metrics.Accuracy(),
        jm.metrics.F1(), # not yet implemented 😅, coming soon?
    ],
    losses=[
        jm.losses.Crossentropy(),
        jm.regularizers.L2(1e-4),
    ],
)

# same API
lms = lms.reset()
# same API
lms = lms.update(target=y, preds=logits, parameters=params)
# compute new returns a dict
lms.compute() # {'accuracy': 0.95, 'f1': 0.87, 'crossentropy': 0.23, 'l2': 0.005}
# same as compute_logs in the case
lms.compute_logs() # {'accuracy': 0.95, 'f1': 0.87, 'crossentropy': 0.23, 'l2': 0.005}
# you can also compute the total loss
total_loss = lms.total_loss() # 0.235
```

```python
def loss_fn(...):
    ...
    batch_updates = lms.batch_updates(target=y, preds=logits, parameters=params)
    loss = batch_updates.total_loss()
    lms = lms.merge(batch_updates)
    ...
    return loss, lms
```

```python
def loss_fn(...):
    ...
    loss, lms = lms.loss_and_update(target=y, preds=logits, parameters=params)
    ...
    return loss, lms
```

```python
def loss_fn(...):
    ...
    batch_updates = lms.batch_updates(target=y, preds=logits, parameters=params)
    loss = batch_updates.total_loss()
    batch_updates = jax.lax.all_gather(batch_updates, "device").aggregate()
    lms = lms.merge(batch_updates)
    ...
    return loss, lms
```