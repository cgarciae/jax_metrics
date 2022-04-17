import jax
import jax.numpy as jnp
import numpy as np
import pytest

import metrix as mtx
from metrix import losses


class TestLosses:
    def test_list(self):

        N = 0

        @jax.jit
        def f(m: mtx.metrics.Losses, target, preds):
            nonlocal N
            N += 1
            return m.loss_and_update(target=target, preds=preds)

        losses = mtx.metrics.Losses(
            [
                mtx.losses.MeanSquaredError(),
                mtx.losses.MeanSquaredError(),
            ]
        ).reset()
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
        def f(m: mtx.metrics.Losses, target, preds):
            nonlocal N
            N += 1
            return m.loss_and_update(target=target, preds=preds)

        losses = mtx.metrics.Losses(
            dict(
                a=mtx.losses.MeanSquaredError(),
                b=mtx.losses.MeanSquaredError(),
            )
        )
        target = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]

        loss, losses = f(losses, target, preds)
        assert N == 1
        assert loss, losses.compute() == (
            2.0,
            {
                "a/mean_squared_error_loss": 1.0,
                "b/mean_squared_error_loss": 1.0,
            },
        )

        loss, losses = f(losses, target, preds)
        assert N == 1
        assert loss, losses.compute() == (
            2.0,
            {
                "a/mean_squared_error_loss": 1.0,
                "b/mean_squared_error_loss": 1.0,
            },
        )


class TestAuxLosses:
    def test_basic(self):

        N = 0

        @jax.jit
        def f(aux_losses: mtx.metrics.AuxLosses, value):
            nonlocal N
            N += 1
            loss_logs = {"aux": value}
            return aux_losses.loss_and_update(aux_values=loss_logs)

        loss_logs = {"aux": jnp.array(1.0, jnp.float32)}
        losses = mtx.metrics.AuxLosses().reset(loss_logs)

        value = jnp.array(1.0, jnp.float32)
        loss, losses = f(losses, value)
        assert N == 1
        assert np.isclose(loss, 1.0)
        assert np.isclose(losses.compute()["aux"], 1.0)

        value = jnp.array(0.0, jnp.float32)
        loss, losses = f(losses, value)

        assert N == 1
        assert np.isclose(loss, 0.0)
        assert np.isclose(losses.compute()["aux"], 0.5)

    def test_named(self):

        N = 0

        @jax.jit
        def f(aux_losses: mtx.metrics.AuxLosses, value):
            nonlocal N
            N += 1
            loss_logs = {"my_loss": value}
            return aux_losses.loss_and_update(aux_values=loss_logs)

        loss_logs = {"my_loss": jnp.array(0.0, jnp.float32)}
        losses = mtx.metrics.AuxLosses().reset(loss_logs)

        value = jnp.array(1.0, jnp.float32)
        loss, losses = f(losses, value)
        assert N == 1
        assert np.isclose(loss, 1.0)
        assert np.isclose(losses.compute()["my_loss"], 1.0)

        value = jnp.array(0.0, jnp.float32)
        loss, losses = f(losses, value)

        assert N == 1
        assert np.isclose(loss, 0.0)
        assert np.isclose(losses.compute()["my_loss"], 0.5)
