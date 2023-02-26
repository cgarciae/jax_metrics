import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_metrics as jm
from jax_metrics import losses


class TestLosses:
    def test_list(self):
        N = 0

        @jax.jit
        def f(m: jm.metrics.Losses, target, preds):
            nonlocal N
            N += 1
            return m.loss_and_update(target=target, preds=preds)

        losses = jm.metrics.Losses(
            dict(
                mean_squared_error=jm.losses.MeanSquaredError(),
                mean_squared_error2=jm.losses.MeanSquaredError(),
            )
        )
        target = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]

        loss, losses = f(losses, target, preds)
        assert N == 1
        loss = 2.0
        assert losses.compute() == {
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
        }

        loss, losses = f(losses, target, preds)
        assert N == 1
        loss = 2.0
        assert losses.compute() == {
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
        }

    def test_dict(self):
        N = 0

        @jax.jit
        def f(m: jm.metrics.Losses, target, preds):
            nonlocal N
            N += 1
            return m.loss_and_update(target=target, preds=preds)

        losses = jm.metrics.Losses(
            dict(
                a=jm.losses.MeanSquaredError(),
                b=jm.losses.MeanSquaredError(),
            )
        )
        target = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]

        loss, losses = f(losses, target, preds)
        assert N == 1
        assert (loss, losses.compute()) == (2.0, {"a": 1.0, "b": 1.0})

        loss, losses = f(losses, target, preds)
        assert N == 1
        assert (loss, losses.compute()) == (2.0, {"a": 1.0, "b": 1.0})
