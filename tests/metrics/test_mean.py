import jax
import jax.numpy as jnp

import jax_metrics as jm


class TestMean:
    def test_from_argument(self):
        metric = jm.metrics.Mean().from_argument("loss")

        metric = metric.update(loss=jnp.array(1.0))
        assert metric.compute() == 1.0

        metric = metric.update(loss=jnp.array(2.0))
        assert metric.compute() == 1.5

        assert metric.metric.total == 3.0
        assert metric.metric.count == 2
