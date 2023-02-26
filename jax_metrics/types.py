import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import typing_extensions as tpe
from simple_pytree import field, static_field

A = tp.TypeVar("A")
B = tp.TypeVar("B")


IndexLike = tp.Union[str, int, tp.Sequence[tp.Union[str, int]]]
PathLike = tp.Tuple[IndexLike, ...]
ScalarLike = tp.Union[float, np.ndarray, jax.Array]


# -----------------------------------------
# Constants
# -----------------------------------------
EPSILON = 1e-7


@jax.tree_util.register_pytree_node_class
@dataclass
class Named(tp.Generic[A]):
    name: str
    value: A

    def tree_flatten(self):
        return (self.value,), self.name

    @classmethod
    def tree_unflatten(cls, name, children):
        return cls(name, children[0])


# -----------------------------------------
# errors
# -----------------------------------------


class OptionalDependencyNotFound(Exception):
    pass
