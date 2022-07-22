import jax
import jax.numpy as jnp
import pytest

from jax_metrics.metrics.f_beta import F1, FBeta


class TestFBeta:
    def test_fbeta(self):
        @jax.jit
        def f(m, target, preds):
            return m.update(target=target, preds=preds)

        target = jnp.asarray([0, 1, 2, 0, 1, 2])
        preds = jnp.asarray([0, 2, 1, 0, 0, 1])
        metric = FBeta(num_classes=3, beta=0.5, average="micro").reset()
        metric = f(metric, target, preds)
        assert jnp.isclose(metric.compute(), 1 / 3)

    def test_f1(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metric = F1(num_classes=10, mdmc_average="global").reset()
        target = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        preds = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])

        metric = f(metric, target, preds)
        assert N == 1
        assert jnp.isclose(metric.compute(), 0.8)

        metric = f(metric, target, preds)
        assert N == 1
        assert jnp.isclose(metric.compute(), 0.8)

        metric = F1(num_classes=3, mdmc_average="global").reset()
        target = jnp.array([0, 1, 2, 0, 1, 2])
        preds = jnp.array([0, 2, 1, 0, 0, 1])
        metric = f(metric, target, preds)
        assert jnp.isclose(metric.compute(), 1 / 3)

        metric = F1(num_classes=3, ignore_index=0, average="macro").reset()
        target = jnp.array([0, 1, 2, 0, 1, 2])
        preds = jnp.array([0, 2, 2, 0, 0, 2])
        metric = f(metric, target, preds)
        assert jnp.isclose(metric.compute(), 0.4)

        metric = F1(num_classes=3, average="macro").reset()
        target = jnp.array([0, 1, 2, 0, 1, 2])
        preds = jnp.array([0, 2, 2, 0, 0, 2])
        metric = f(metric, target, preds)
        assert jnp.isclose(metric.compute(), 0.5333336)

    def test_logits_preds(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metric = F1(num_classes=2, mdmc_average="global").reset()
        target = jnp.array([0, 0, 1, 1, 1])
        preds = jnp.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        metric = f(metric, target, preds)
        assert N == 1
        assert jnp.isclose(metric.compute(), 0.2)

        metric = F1(num_classes=2, ignore_index=0, average="macro").reset()
        metric = f(metric, target, preds)
        assert jnp.isclose(metric.compute(), 1 / 3)
