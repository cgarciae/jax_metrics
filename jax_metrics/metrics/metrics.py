import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from simple_pytree import field, static_field

from jax_metrics import types, utils
from jax_metrics.metrics.metric import Metric, SumMetric

M = tp.TypeVar("M", bound="Metrics")
A = tp.TypeVar("A", bound="AuxMetrics")


@dataclasses.dataclass
class Metrics(Metric):
    metrics: tp.Dict[str, Metric]

    def reset(self: M) -> M:
        metrics = {name: metric.reset() for name, metric in self.metrics.items()}
        return self.replace(metrics=metrics)

    def update(self: M, **kwargs) -> M:
        """
        Update all metrics with the given values. Each metric will receive the
        same keyword arguments but each can internally select the values to use.
        If a required value is not provided, the metric will raise a TypeError.

        Arguments:
            **kwargs: Keyword arguments to pass to each metric.

        Returns:
            Metrics instance with updated state.
        """
        metrics = {
            name: metric.update(**kwargs) for name, metric in self.metrics.items()
        }
        return self.replace(metrics=metrics)

    def compute(self) -> tp.Dict[str, jax.Array]:
        outputs = {}
        names = set()

        for name, metric in self.metrics.items():
            value = metric.compute()

            for path, value, parent_iterable in utils._flatten_names(value):
                name = utils._unique_name(names, name)

                if path:
                    if parent_iterable:
                        name = f"{path}/{name}"
                    else:
                        name = path

                outputs[name] = value

        return outputs

    def slice(self, **kwargs: types.IndexLike) -> "Metrics":
        metrics = {
            name: metric.index_into(**kwargs) for name, metric in self.metrics.items()
        }
        return self.replace(metrics=metrics)

    def merge(self: M, other: M) -> M:
        return type(self)(
            metrics={
                name: metric.merge(other.metrics[name])
                for name, metric in self.metrics.items()
            }
        )

    def reduce(self: M) -> M:
        return type(self)(
            metrics={name: metric.reduce() for name, metric in self.metrics.items()}
        )


class AuxMetrics(SumMetric):
    totals: tp.Optional[tp.Dict[str, jax.Array]] = field()
    counts: tp.Optional[tp.Dict[str, jax.Array]] = field()
    names: tp.Tuple[str, ...] = static_field()

    def __init__(self, names: tp.Iterable[str]):
        self.names = tuple(names)
        self.__dict__.update(self._initial_values())

    def _initial_values(self: A) -> tp.Dict[str, tp.Any]:
        totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.names}
        counts = {name: jnp.array(0, dtype=jnp.uint32) for name in self.names}

        return dict(totals=totals, counts=counts)

    def reset(self: A) -> A:
        return self.replace(**self._initial_values())

    def update(self: A, aux_values: tp.Dict[str, jax.Array], **_) -> A:
        if self.totals is None or self.counts is None:
            raise ValueError("AuxMetrics not initialized, call 'reset()' first")

        totals = {
            name: (self.totals[name] + aux_values[name]).astype(self.totals[name].dtype)
            for name in self.totals
        }
        counts = {
            name: (self.counts[name] + np.prod(aux_values[name].shape)).astype(
                self.counts[name].dtype
            )
            for name in self.counts
        }

        return self.replace(totals=totals, counts=counts)

    def compute(self) -> tp.Dict[str, jax.Array]:
        if self.totals is None or self.counts is None:
            raise ValueError("AuxMetrics not initialized, call `reset()` first")

        return {name: self.totals[name] / self.counts[name] for name in self.totals}

    def compute_logs(self) -> tp.Dict[str, jax.Array]:
        return self.compute()

    def __call__(self: A, aux_values: tp.Any) -> tp.Tuple[tp.Dict[str, jax.Array], A]:
        return super().__call__(aux_values=aux_values)
