import typing as tp
from abc import abstractmethod

import jax
import jax.numpy as jnp
from simple_pytree import Pytree, field, static_field

from jax_metrics import types

M = tp.TypeVar("M", bound="Metric")
MA = tp.TypeVar("MA", bound="MapArgsMetric")
Slice = tp.Tuple[tp.Union[int, str], ...]


class Metric(Pytree):
    """
    Encapsulates metric logic and state. Metrics accumulate state between calls such
    that their output value reflect the metric as if calculated on the whole data
    given up to that point.
    """

    def __call__(self: M, **kwargs: tp.Any) -> tp.Tuple[tp.Any, M]:
        metric: M = self

        batch_updates = metric.batch_updates(**kwargs)
        batch_values = batch_updates.compute()

        metric = metric.merge(batch_updates)

        return batch_values, metric

    @abstractmethod
    def reset(self: M) -> M:
        """
        Resets the metric state.

        Returns:
            Metric with the initial state
        """
        ...

    @abstractmethod
    def update(self: M, **kwargs: tp.Any) -> M:
        """
        Update the metric with the given data. Each metric accepts a different set of
        keyword arguments and must accept other keyword arguments, even if they not used
        by as remaining `**kwargs`.

        Arguments:
            **kwargs: data to update the metric with

        Returns:
            Metric with updated state
        """
        ...

    @abstractmethod
    def compute(self) -> tp.Any:
        """
        Compute the current metric value.
        """
        ...

    @abstractmethod
    def reduce(self: M) -> M:
        """
        Aggregate metric state. It assumes the metric's internal state has an additional
        'device' dimension on the 0th axis.

        Example:

        ```python
        batch_updates = metric.batch_updates(**kwargs)
        batch_updates = jax.lax.all_gather(batch_updates, axis_name="device")
        batch_updates = batch_updates.reduce()

        metric = metric.merge(batch_updates)
        ```

        Returns:
            Metric with aggregated state
        """
        # return jax.tree_map(lambda x: jnp.sum(x, axis=0), self)
        ...

    @abstractmethod
    def merge(self: M, other: M) -> M:
        """
        Merge the state of two metrics of the same type. Usually used to merge
        a metric with its batch_updates.

        Example:

        ```python
        batch_updates = metric.batch_updates(**kwargs)
        metric = metric.merge(batch_updates)
        ```
        """
        # return jax.tree_map(lambda x, y: x + y, self, other)
        ...

    def batch_updates(self: M, **kwargs: tp.Any) -> M:
        """
        Compute metric updates for a batch of data. Equivalent to
        `.reset().update(**kwargs)`.

        Arguments:
            kwargs: data to update the metric with

        Returns:
            Metric with updated state
        """
        return self.reset().update(**kwargs)

    def index_into(self, **kwargs: types.IndexLike) -> "IndexedMetric":
        """
        Returns a metric that "indexes" the specified keyword arguments expected by
        `.update()`. You can index into nested structures such as combinations of lists,
        tuples, dicts, or any other structure that supports indexing (`__getitem__`).

        Example:

        ```python
        metrics = jm.Metrics([
            jm.metrics.Mean().index_into(values=["a"]),
            jm.metrics.Mean().index_into(values=["b"]),
        ]).reset()


        metrics = metrics.update(values={
            "a": loss0,
            "b": loss1,
        })
        ```

        Here `values` is set to a dict of arrays, but thanks to `.index_into()`
        each loss can index into its correspoding array. This also works with


        Arguments:
            **kwargs: keyword arguments to be indexed

        Returns:
            A IndexedMetric instance
        """
        return IndexedMetric(self, kwargs)

    def map_arg(self, **kwargs: str) -> "MapArgsMetric":
        """
        Returns a metric that renames the keyword arguments expected by `.update()`.

        Example:

        ```python
        mean = jm.metrics.Mean().map_arg(values="loss").reset()
        ...
        loss = loss_fn(x, y)
        mean = mean.update(loss=loss)
        ```

        Arguments:
            **kwargs: keyword arguments to be renamed

        Returns:
            A MapArgsMetric instance
        """
        return MapArgsMetric(self, kwargs)


class SumMetric(Metric):
    def merge(self: M, other: M) -> M:
        return jax.tree_map(lambda x, y: x + y, self, other)

    def reduce(self: M) -> M:
        return jax.tree_map(lambda x: jnp.sum(x, axis=0), self)


class IndexedMetric(Metric):
    metric: Metric = field()
    arg_slice: tp.Dict[str, Slice] = static_field()

    def __init__(
        self,
        metric: Metric,
        arg_slice: tp.Dict[str, types.IndexLike],
    ):
        self.metric = metric
        self.arg_slice = {
            key: (index,) if isinstance(index, (int, str)) else tuple(index)
            for key, index in arg_slice.items()
        }

    def reset(self) -> "IndexedMetric":
        return self.replace(metric=self.metric.reset())

    def reduce(self) -> "IndexedMetric":
        return self.replace(metric=self.metric.reduce())

    def merge(self, other: Metric) -> "IndexedMetric":
        if not isinstance(other, IndexedMetric):
            raise ValueError(
                f"Can only merge IndexedMetric with IndexedMetric, got {type(other)}"
            )
        return self.replace(metric=self.metric.merge(other.metric))

    def update(self, **kwargs: tp.Any) -> "IndexedMetric":
        # slice the arguments
        for key, slices in self.arg_slice.items():
            for index in slices:
                kwargs[key] = kwargs[key][index]

        return self.replace(metric=self.metric.update(**kwargs))

    def compute(self) -> tp.Any:
        return self.metric.compute()


class MapArgsMetric(Metric):
    metric: Metric = field()
    args_map: tp.Dict[str, str] = static_field()

    def __init__(self, metric: Metric, args_map: tp.Dict[str, str]):
        self.metric = metric
        self.args_map = args_map

    def reset(self: MA) -> MA:
        return self.replace(metric=self.metric.reset())

    def update(self: MA, **kwargs: tp.Any) -> MA:
        for arg in self.args_map:
            if arg not in kwargs:
                raise KeyError(f"'{arg}' expected but not given")

        kwarg_updates = {
            next_arg: kwargs[prev_arg] for prev_arg, next_arg in self.args_map.items()
        }

        # delete previous kwargs
        for arg in self.args_map:
            del kwargs[arg]

        # add new kwargs
        kwargs.update(kwarg_updates)

        return self.replace(metric=self.metric.update(**kwargs))

    def compute(self) -> tp.Any:
        return self.metric.compute()
