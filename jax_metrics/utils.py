import typing as tp

import jax
import jax.numpy as jnp
from treeo.utils import _get_name, _lower_snake_case, _unique_name, _unique_names

from jax_metrics import types


def _flatten_names(inputs: tp.Any) -> tp.List[tp.Tuple[str, tp.Any, bool]]:
    return [
        ("/".join(map(str, path)), value, parent_iterable)
        for path, value, parent_iterable in _flatten_names_helper((), inputs, True)
    ]


def _flatten_names_helper(
    path: types.PathLike, inputs: tp.Any, parent_iterable: bool
) -> tp.Iterable[tp.Tuple[types.PathLike, tp.Any, bool]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names_helper(path, value, True)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _flatten_names_helper(path + (name,), value, False)
    else:
        yield (path, inputs, parent_iterable)
