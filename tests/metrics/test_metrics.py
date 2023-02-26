import jax
import jax.numpy as jnp
import pytest

import jax_metrics as jm
from jax_metrics import metrics


class TestAccuracy:
    def test_list(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metrics = jm.metrics.Metrics(
            {
                "accuracy": jm.metrics.Accuracy(num_classes=10),
                "accuracy2": jm.metrics.Accuracy(num_classes=10),
            }
        )
        target = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        preds = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metrics = f(metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {"accuracy": 0.8, "accuracy2": 0.8}

        metrics = f(metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {"accuracy": 0.8, "accuracy2": 0.8}

    def test_dict(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metrics = jm.metrics.Metrics(
            dict(
                a=jm.metrics.Accuracy(num_classes=10),
                b=jm.metrics.Accuracy(num_classes=10),
            )
        )
        target = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        preds = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metrics = f(metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {"a": 0.8, "b": 0.8}

        metrics = f(metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {"a": 0.8, "b": 0.8}


class TestAuxMetrics:
    def test_basic(self):
        N = 0

        @jax.jit
        def f(aux_metrics: jm.AuxMetrics, value):
            nonlocal N
            N += 1
            metric_logs = {"my_metric": value}
            return aux_metrics.update(aux_values=metric_logs)

        metric_logs = {"my_metric": jnp.array(0.0, jnp.float32)}
        metrics: jm.AuxMetrics = jm.AuxMetrics(names=metric_logs)

        value = jnp.array(1.0, jnp.float32)
        metrics = f(metrics, value)
        assert N == 1
        assert metrics.compute() == {"my_metric": 1.0}

        value = jnp.array(0.0, jnp.float32)
        metrics = f(metrics, value)

        assert N == 1
        assert metrics.compute() == {"my_metric": 0.5}

    def test_named(self):
        N = 0

        @jax.jit
        def f(aux_metrics: jm.AuxMetrics, value: jnp.ndarray):
            nonlocal N
            N += 1
            metric_logs = {"my_metric": value}
            return aux_metrics.update(aux_values=metric_logs)

        metric_logs = {"my_metric": jnp.array(0.0, jnp.float32)}
        metrics: jm.AuxMetrics = jm.AuxMetrics(names=metric_logs)

        value = jnp.array(1.0, jnp.float32)
        metrics = f(metrics, value)
        assert N == 1
        assert metrics.compute() == {"my_metric": 1.0}

        value = jnp.array(0.0, jnp.float32)
        metrics = f(metrics, value)

        assert N == 1
        assert metrics.compute() == {"my_metric": 0.5}
