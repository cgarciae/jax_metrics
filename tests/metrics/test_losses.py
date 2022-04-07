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
            [
                jm.losses.MeanSquaredError(),
                jm.losses.MeanSquaredError(),
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
        class MyModule(jm.Module):
            aux: jnp.ndarray = jm.LossLog.node()
            some_value: jnp.ndarray = to.node()

            def __init__(self) -> None:
                self.aux = jnp.array(1.0, jnp.float32)
                self.some_value = jnp.array(10.0, jnp.float32)

        N = 0

        @jax.jit
        def f(module: MyModule, aux_losses: jm.metrics.AuxLosses):
            nonlocal N
            N += 1
            loss_logs = module.filter(jm.LossLog).as_logs()
            return aux_losses.loss_and_update(aux_values=loss_logs)

        module = MyModule()

        loss_logs = module.filter(jm.LossLog).as_logs()
        losses = jm.metrics.AuxLosses().reset(loss_logs)

        loss, losses = f(module, losses)
        assert N == 1
        assert np.isclose(loss, 1.0)
        assert np.isclose(losses.compute()["aux"], 1.0)

        module = module.replace(aux=jnp.array(0.0, jnp.float32))
        loss, losses = f(module, losses)

        assert N == 1
        assert np.isclose(loss, 0.0)
        assert np.isclose(losses.compute()["aux"], 0.5)

    def test_named(self):
        class MyModule(jm.Module):
            aux: jm.Named[jnp.ndarray] = jm.LossLog.node()
            some_value: jnp.ndarray = to.node()

            def __init__(self) -> None:
                self.aux = jm.Named("my_loss", jnp.array(1.0, jnp.float32))
                self.some_value = jnp.array(10.0, jnp.float32)

        N = 0

        @jax.jit
        def f(module: MyModule, aux_losses: jm.metrics.AuxLosses):
            nonlocal N
            N += 1
            loss_logs = module.filter(jm.LossLog).as_logs()
            return aux_losses.loss_and_update(aux_values=loss_logs)

        module = MyModule()

        loss_logs = module.filter(jm.LossLog).as_logs()
        losses = jm.metrics.AuxLosses().reset(loss_logs)

        loss, losses = f(module, losses)
        assert N == 1
        assert np.isclose(loss, 1.0)
        assert np.isclose(losses.compute()["my_loss"], 1.0)

        module.aux.value = jnp.array(0.0, jnp.float32)
        loss, losses = f(module, losses)

        assert N == 1
        assert np.isclose(loss, 0.0)
        assert np.isclose(losses.compute()["my_loss"], 0.5)
