import typing as tp

import jax
import numpy as np

A = tp.TypeVar("A")
B = tp.TypeVar("B")


IndexLike = tp.Union[str, int, tp.Sequence[tp.Union[str, int]]]
PathLike = tp.Tuple[IndexLike, ...]
ScalarLike = tp.Union[float, np.ndarray, jax.Array]


# -----------------------------------------
# Constants
# -----------------------------------------
EPSILON = 1e-7
