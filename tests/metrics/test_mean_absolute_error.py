import hypothesis as hp
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics as tm
from hypothesis import strategies as st

import jax_metrics as jm


class TestMAE:
    def test_mae_basic(self):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mae_tx = jm.metrics.MeanAbsoluteError()
        mae_tx_value, mae_tx = mae_tx(target=target, preds=preds)

        mae_tm = tm.MeanAbsoluteError()
        mae_tm_value = mae_tm(torch.from_numpy(preds), torch.from_numpy(target))
        assert np.isclose(np.array(mae_tx_value), mae_tm_value.numpy())

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mae_weights_batch_dim(self, use_sample_weight):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        if use_sample_weight:
            sum = 0
            while sum == 0:
                sample_weight = np.random.choice([0, 1], 8)
                sum = sample_weight.sum()

        params = {"target": target, "preds": preds}
        mae_tx = jm.metrics.MeanAbsoluteError().reset()
        if use_sample_weight:
            params.update({"sample_weight": sample_weight})
        mae_tx_value, mae_tx = mae_tx(**params)

        mae_tx = jm.metrics.MeanAbsoluteError().reset()
        if use_sample_weight:
            target, preds = target[sample_weight == 1], preds[sample_weight == 1]
        mae_tx_no_sample_weight, mae_tx = mae_tx(target=target, preds=preds)

        assert np.isclose(mae_tx_value, mae_tx_no_sample_weight)

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mae_weights_values_dim(self, use_sample_weight):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        params = {"target": target, "preds": preds}
        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8 * 20).reshape((8, 20))
            params.update({"sample_weight": sample_weight})

        mae_tx, _ = jm.metrics.MeanAbsoluteError().reset()(**params)

        assert isinstance(mae_tx, jax.Array)

    def test_accumulative_mae(self):
        mae_tx = jm.metrics.MeanAbsoluteError().reset()
        mae_tm = tm.MeanAbsoluteError()
        for batch in range(2):
            target = np.random.randn(8, 5, 5)
            preds = np.random.randn(8, 5, 5)

            mae_tx = mae_tx.update(target=target, preds=preds)
            mae_tm(torch.from_numpy(preds), torch.from_numpy(target))

        assert np.isclose(
            np.array(mae_tx.compute()),
            mae_tm.compute().numpy(),
        )

    def test_mae_short(self):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mae_tx_long, _ = jm.metrics.MeanAbsoluteError().reset()(
            target=target, preds=preds
        )
        mae_tx_short, _ = jm.metrics.MAE().reset()(target=target, preds=preds)
        assert np.isclose(np.array(mae_tx_long), np.array(mae_tx_short))
