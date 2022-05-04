import jax
import jax.numpy as jnp
import pytest

import jax_metrics as jm
from jax_metrics import losses


class TestLossAndLogs:
    def test_basic(self):

        N = 0

        @jax.jit
        def f(
            metrics: jm.metrics.LossesAndMetrics,
            target,
            preds,
            aux_loss,
            aux_metric,
        ):
            nonlocal N
            N += 1
            return metrics.update(
                target=target,
                preds=preds,
                aux_losses={"aux_loss": aux_loss},
                aux_metrics={"aux_metric": aux_metric},
            )

        metrics = jm.metrics.LossesAndMetrics(
            losses=jm.metrics.Losses(
                [
                    jm.losses.MeanSquaredError(),
                    jm.losses.MeanSquaredError(),
                ]
            ).slice(target="losses", preds="losses"),
            metrics=jm.metrics.Metrics(
                dict(
                    a=jm.metrics.Accuracy(num_classes=10),
                    b=jm.metrics.Accuracy(num_classes=10),
                )
            ).slice(target="metrics", preds="metrics"),
            aux_losses=jm.metrics.AuxLosses(),
            aux_metrics=jm.metrics.AuxMetrics(),
        ).init(
            aux_losses={"aux_loss": jnp.array(0.0, jnp.float32)},
            aux_metrics={"aux_metric": jnp.array(0.0, jnp.float32)},
        )
        target = dict(
            losses=jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :],
        )
        preds = dict(
            losses=jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :],
        )

        aux_loss = jnp.array(1.0, jnp.float32)
        aux_metric = jnp.array(2.0, jnp.float32)

        metrics = f(metrics, target, preds, aux_loss, aux_metric)
        assert N == 1
        assert metrics.compute() == {
            "loss": 3.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 1.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 2.0,
        }

        aux_loss = jnp.array(3.0, jnp.float32)
        aux_metric = jnp.array(4.0, jnp.float32)

        metrics = f(metrics, target, preds, aux_loss, aux_metric)
        assert N == 1
        assert metrics.compute() == {
            "loss": 4.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 2.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 3.0,
        }

    def test_batch_loss(self):
        N = 0

        @jax.jit
        def f(
            metrics: jm.metrics.LossesAndMetrics,
            target,
            preds,
            aux_loss,
            aux_metric,
        ):
            nonlocal N
            N += 1
            return metrics.loss_and_update(
                target=target,
                preds=preds,
                aux_losses={"aux_loss": aux_loss},
                aux_metrics={"aux_metric": aux_metric},
            )

        metrics = jm.metrics.LossesAndMetrics(
            losses=jm.metrics.Losses(
                [
                    jm.losses.MeanSquaredError(),
                    jm.losses.MeanSquaredError(),
                ]
            ).slice(target="losses", preds="losses"),
            metrics=jm.metrics.Metrics(
                dict(
                    a=jm.metrics.Accuracy(num_classes=10),
                    b=jm.metrics.Accuracy(num_classes=10),
                )
            ).slice(target="metrics", preds="metrics"),
            aux_losses=jm.metrics.AuxLosses(),
            aux_metrics=jm.metrics.AuxMetrics(),
        ).init(
            aux_losses={"aux_loss": jnp.array(0.0, jnp.float32)},
            aux_metrics={"aux_metric": jnp.array(0.0, jnp.float32)},
        )
        target = dict(
            losses=jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :],
        )
        preds = dict(
            losses=jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :],
        )

        aux_loss = jnp.array(1.0, jnp.float32)
        aux_metric = jnp.array(2.0, jnp.float32)

        loss, metrics = f(metrics, target, preds, aux_loss, aux_metric)
        assert N == 1
        assert loss == 3.0
        assert metrics.compute() == {
            "loss": 3.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 1.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 2.0,
        }

        aux_loss = jnp.array(3.0, jnp.float32)
        aux_metric = jnp.array(4.0, jnp.float32)

        loss, metrics = f(metrics, target, preds, aux_loss, aux_metric)
        assert N == 1
        assert loss == 5.0
        assert metrics.compute() == {
            "loss": 4.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 2.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 3.0,
        }
