from typing import Optional, Union

import jax.numpy as jnp

from jax_metrics.metrics import utils as metric_utils
from jax_metrics.metrics.accuracy import Accuracy
from jax_metrics.metrics.utils import AverageMethod, DataType, MDMCAverageMethod


class FBeta(Accuracy):
    r"""Computes `F-score`, ported from [torchmetrics](https://github.com/PytorchLightning/metrics)

    Accepts logit scores or probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Call accepts
    - ``preds`` : ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` : ``(N, ...)``

    If preds and target are the same shape and preds is a float array, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label logits and probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Arguments:
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        beta:
            Beta coefficient in the F measure.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:
            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).
            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.
            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.
        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:
            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.
            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.
            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The default value (``None``) will be interpreted
            as 1 for these inputs.
            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.
    Raises:
        ValueError:
        If `top_k` is not an `integer` larger than `0`.
    ValueError:
        If `average` is none of `"micro"`, `"macro"`, `"weighted"`, `"samples"`, `"none"`, `None`.
    ValueError:
        If two different input modes are provided, eg. using `multi-label` with `multi-class`.
    ValueError:
        If `top_k` parameter is set for `multi-label` inputs.

    Example:
        >>> import jax.numpy as jnp
        >>> from jax_metrics.metrics.f_beta import FBeta
        >>> target = jnp.asarray([0, 1, 2, 0, 1, 2])
        >>> preds = jnp.asarray([0, 2, 1, 0, 0, 1])
        >>> f_beta = FBeta(num_classes=3, beta=0.5)
        >>> f_beta(preds, target)
        tensor(0.3333)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        beta: float = 1.0,
        average: Union[str, AverageMethod] = AverageMethod.MICRO,
        mdmc_average: Union[str, MDMCAverageMethod] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        mode: DataType = DataType.MULTICLASS,
        name: Optional[str] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        super().__init__(
            threshold=threshold,
            num_classes=num_classes,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass,
            mode=mode,
            name=name,
            dtype=dtype,
        )
        self.beta = beta

    def compute(self) -> jnp.ndarray:
        """
        Computes f-beta based on inputs passed in to `update` previously.

        Returns:
            F-beta score
        """
        if self.tp is None or self.fp is None or self.tn is None or self.fn is None:
            raise ValueError(
                "F-Beta metric has not been initialized, call 'reset()' first."
            )

        return metric_utils._fbeta_compute(
            self.tp,
            self.fp,
            self.tn,
            self.fn,
            self.beta,
            self.ignore_index,
            self.average,
            self.mdmc_average,
            self.mode,
        )


class F1(FBeta):
    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        average: Union[str, AverageMethod] = AverageMethod.MICRO,
        mdmc_average: Union[str, None, MDMCAverageMethod] = MDMCAverageMethod.GLOBAL,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        mode: DataType = DataType.MULTICLASS,
        name: Optional[str] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        super().__init__(
            threshold=threshold,
            num_classes=num_classes,
            beta=1.0,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass,
            mode=mode,
            name=name,
            dtype=dtype,
        )
