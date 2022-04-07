# JAX Metrics

_A Pytree Module system for Deep Learning in JAX_

#### Main Features

* 💡 **Intuitive**: Modules contain their own parameters and respect Object Oriented semantics like in PyTorch and Keras.
* 🌳 **Pytree-based**:  Modules are Pytrees whose leaves are its parameters, meaning they are fully compatible with `jit`, `grad`, `vmap`, etc.

JAX Metrics is implemented on top of [Treeo](https://github.com/cgarciae/treeo) and reexports all of its API for convenience.

[Getting Started](#getting-started) | [User Guide](https://cgarciae.github.io/jax_metrics/user-guide/intro) | [Examples](#examples) |  [Documentation](https://cgarciae.github.io/jax_metrics)

## What is included?
* A base `Module` class.
* A `nn` module for with common layers implemented as wrappers over Flax layers.
* A `losses` module with common loss functions.
* A `metrics` module with common metrics.
* An `Optimizer` class that can wrap any optax optimizer.
## Why JAX Metrics?
<details>
<summary>Show</summary><br>

Despite all JAX benefits, current Module systems are not intuitive to new users and add additional complexity not present in frameworks like PyTorch or Keras. JAX Metrics takes inspiration from S4TF and delivers an intuitive experience using JAX Pytree infrastructure.

<details>
<summary>Current Alternative's Drawbacks and Solutions</summary>

Currently we have many alternatives like Flax, Haiku, Objax, that have one or more of the following drawbacks:

* Module structure and parameter structure are separate, and parameters have to be manipulated around by the end-user, which is not intuitive. In JAX Metrics, parameters are stored in the modules themselves and can be accessed directly.
* Monadic architecture adds complexity. Flax and Haiku use an `apply` method to call modules that set a context with parameters, rng, and different metadata, which adds additional overhead to the API and creates an asymmetry in how Modules are being used inside and outside a context. In JAX Metrics, modules can be called directly.
* Among different frameworks, parameter surgery requires special consideration and is challenging to implement. Consider a standard workflow such as transfer learning, transferring parameters and state from a  pre-trained module or submodule as part of a new module; in different frameworks, we have to know precisely how to extract their parameters and how to insert them into the new parameter structure/dictionaries such that it is in agreement with the new module structure. In JAX Metrics, just as in PyTorch / Keras, we enable to pass the (sub)module to the new module, and parameters are automatically added to the new structure.
* Multiple frameworks deviate from JAX semantics and require particular versions of `jit`, `grad`, `vmap`, etc., which makes it harder to integrate with other JAX libraries. JAX Metrics's Modules are plain old JAX PyTrees and are compatible with any JAX library that supports them.
* Other Pytree-based approaches like Parallax and Equinox do not have a total state management solution to handle complex states as encountered in Flax. JAX Metrics has the Filter and Update API, which is very expressive and can effectively handle systems with a complex state.

</details>
</details>

## Installation
Install using pip:
```bash
pip install jax_metrics
```


## Getting Started
<!-- Remake Getting Started now that most content is in the User Guide -->

This is a small appetizer to give you a feel for how using JAX Metrics looks like, be sure to checkout the [User Guide](https://cgarciae.github.io/jax_metrics/user-guide/intro) for a more in-depth explanation.
```python
import jax_metrics as jm
import numpy as np
import jax, optax


# create some data
x = np.random.uniform(size=(50, 1))
y = 1.3 * x ** 2 - 0.3 + np.random.normal(size=x.shape)



# initialize a Module, its simple
model = jm.MLP([64, 1]).init(key=42, inputs=x)
# define an optimizer, init with model params
optimizer = jm.Optimizer(optax.adam(4e-3)).init(model)



# define loss function, notice
# Modules are jit-abel and differentiable 🤯
@jax.jit
@jax.grad
def loss_fn(model: jm.MLP, x, y):
    # forward is a simple call
    preds = model(x)
    # MSE
    return ((preds - y) ** 2).mean()



# basic training loop
for step in range(500):

    # grads have the same type as model
    grads: jm.MLP = loss_fn(model, x, y)
    # apply the gradient updates
    model = optimizer.update(grads, model)



# Pytorch-like eval mode
model = model.eval()
preds = model(x)
```
#### Custom Modules
<details>
<summary>Show</summary><br>

Modules are Treeo `Tree`s, which are Pytrees. When creating core layers you often mark fields that will contain state that JAX should be aware as `nodes` by assigning class variables to the output of functions like `jm.Parameter.node()`:

```python
import jax_metrics as jm

class Linear(jm.Module):
    # use Treeo's API to define Parameter nodes
    w: jnp.ndarray = jm.Parameter.node()
    b: jnp.ndarray = jm.Parameter.node()

    def __init__(self, features_out: int):
        self.features_out = features_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # init will call forward, we can know if we are inside it
        if self.initializing():
            # `next_key` only available during `init`
            key = jm.next_key() 
            # leverage shape inference
            self.w = jax.random.uniform(
                key, shape=[x.shape[-1], self.features_out]
            )
            self.b = jnp.zeros(shape=[self.features_out])

        # linear forward
        return jnp.dot(x, self.w) + self.b

model = Linear(10).init(key=42, inputs=x)
```
Node field types (e.g. `jm.Parameter`) are called Kinds and JAX Metrics exports a whole family of Kinds which serve for differente purposes such as holding non-differentiable state (`jm.BatchStats`), metric's state (`jm.MetricState`), logging, etc. Checkout the [kinds](https://cgarciae.github.io/jax_metrics/user-guide/kinds) section for more information.
</details>

#### Composite Modules
<details>
<summary>Show</summary><br>

Composite Modules usually hold and call other Modules within them, while they would be instantiate inside `__init__` and used later in `__call__` like in Pytorch / Keras, in JAX Metrics you usually leverage the `@jm.compact` decorator over the `__call__` method to define the submodules inline.
```python
class MLP(jm.Module):
    def __init__(self, features: Sequence[int]):
        self.features = features

    # compact lets you define submodules on the fly
    @jm.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for units in self.features[:-1]:
            x = Linear(units)(x)
            x = jax.nn.relu(x)

        return Linear(self.features[-1])(x)

model = MLP([32, 10]).init(key=42, inputs=x)
```
Under the hood all calls to submodule constructors (e.g. `Linear(...)`) inside `compact` are assigned to fields in the parent Module (`MLP`) so they are part of the same Pytree, their field names are available under the `._subtrees` attribute. `compact` must always define submodules in the same order.

</details>

## Status
JAX Metrics is in an early stage, things might break between versions but we will respect semanting versioning. Since JAX Metrics layers are numerically equivalent to Flax, it borrows some maturity and yields more confidence over its results. Feedback is much appreciated.

**Roadmap**:

- Wrap all Flax Linen Modules
- Implement more layers, losses, and metrics.
- Create applications and pretrained Modules.

Contributions are welcomed!

## Sponsors 💚
* [Quansight](https://www.quansight.com) - paid development time

## Examples
Checkout the [/examples](examples) directory for more detailed examples. Here are a few additional toy examples:


#### Linear Regression
This is a simple but realistic example of how JAX Metrics is used.

```python
from functools import partial
from typing import Union
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import jax_metrics as jm

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))


# differentiate only w.r.t. parameters
def loss_fn(params, model, x, y):
    # merge params into model
    model = model.merge(params)

    preds = model(x)
    loss = jnp.mean((preds - y) ** 2)

    # the model may contain state updates
    # so it should be returned
    return loss, model


grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

# both model and optimizer are jit-able
@jax.jit
def train_step(model, x, y, optimizer):
    # select only the parameters
    params = model.parameters()

    (loss, model), grads = grad_fn(params, model, x, y)

    # update params and model
    params = optimizer.update(grads, params)
    model = model.merge(params)

    # return new model and optimizer
    return loss, model, optimizer


model = jm.Linear(1).init(42, x)
optimizer = jm.Optimizer(optax.adam(0.01)).init(model)

for step in range(300):
    loss, model, optimizer = train_step(model, x, y, optimizer)
    if step % 50 == 0:
        print(f"loss: {loss:.4f}")

# eval mode "turns off" layers like Dropout / BatchNorm
model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
preds = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, preds, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
```

#### A Stateful Module
Here is an example of creating a stateful module of a `RollingMean` metric and using them with `jax.jit`. For a real use cases use the metrics inside `jax_metrics.metrics`.

```python
class RollingMean(jm.Module):
    count: jnp.ndarray = jm.State.node()
    total: jnp.ndarray = jm.State.node()

    def __init__(self):
        self.count = jnp.array(0, dtype=jnp.int32)
        self.total = jnp.array(0.0, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.count += np.prod(x.shape)
        self.total += x.sum()

        return self.total / self.count

@jax.jit
def update(x: jnp.ndarray, metric: RollingMean) -> Tuple[jnp.ndarray, RollingMean]:
    mean = metric(x)
    return mean, metric # return mean value and updated metric


metric = RollingMean()

for i in range(10):
    x = np.random.uniform(-1, 1, size=(100, 1))
    mean, metric = update(x, metric)

print(mean)
```
