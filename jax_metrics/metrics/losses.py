import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from simple_pytree import field, static_field

from jax_metrics import types, utils
from jax_metrics.losses.loss import Loss
from jax_metrics.metrics.metric import Metric, SumMetric
from jax_metrics.metrics.metrics import AuxMetrics

M = tp.TypeVar("M", bound="Losses")

LossFn = tp.Callable[..., jax.Array]


class Losses(SumMetric):
    totals: tp.Dict[str, jax.Array]
    counts: tp.Dict[str, jax.Array]
    losses: tp.Dict[str, LossFn] = static_field()

    def __init__(
        self,
        losses: tp.Dict[str, LossFn],
    ):
        self.losses = losses
        self.totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.losses}
        self.counts = {name: jnp.array(0, dtype=jnp.uint32) for name in self.losses}

    def reset(self: M) -> M:
        return jax.tree_map(jnp.zeros_like, self)

    def update(self: M, **kwargs) -> M:
        if self.totals is None or self.counts is None:
            raise ValueError("Losses not initialized, call 'reset()' first")

        totals = {
            name: self.totals[name] + loss(**kwargs)
            for name, loss in self.losses.items()
        }
        counts = {name: value + 1 for name, value in self.counts.items()}

        return self.replace(totals=totals, counts=counts)

    def compute(self) -> tp.Dict[str, jax.Array]:
        return {name: self.totals[name] / self.counts[name] for name in self.losses}

    def total_loss(self) -> jax.Array:
        return sum(self.compute().values(), jnp.array(0.0))

    def loss_and_update(self: M, **kwargs) -> tp.Tuple[jax.Array, M]:
        batch_updates = self.batch_updates(**kwargs)
        loss = batch_updates.total_loss()
        metrics = self.merge(batch_updates)

        return loss, metrics
