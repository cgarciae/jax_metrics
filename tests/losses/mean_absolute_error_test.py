import jax
import jax.numpy as jnp

import jax_metrics as jm

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mae = jm.losses.MeanAbsoluteError()

    assert mae(target=target, preds=preds) == 0.5

    # Calling with 'sample_weight'.
    assert mae(target=target, preds=preds, sample_weight=jnp.array([0.7, 0.3])) == 0.25

    # Using 'sum' reduction type.
    mae = jm.losses.MeanAbsoluteError(reduction=jm.losses.Reduction.SUM)

    assert mae(target=target, preds=preds) == 1.0

    # Using 'none' reduction type.
    mae = jm.losses.MeanAbsoluteError(reduction=jm.losses.Reduction.NONE)

    assert list(mae(target=target, preds=preds)) == [0.5, 0.5]


#
def test_function():
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = jm.losses.mean_absolute_error(target, preds)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, jnp.mean(jnp.abs(target - preds), axis=-1))


if __name__ == "__main__":
    test_basic()
    test_function()
