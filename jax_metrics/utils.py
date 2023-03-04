import re
import typing as tp

import jax
import jax.numpy as jnp

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


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _unique_name(
    names: tp.Set[str],
    name: str,
):
    if name in names:
        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: tp.Iterable[str],
    *,
    existing_names: tp.Optional[tp.Set[str]] = None,
) -> tp.Iterable[str]:
    if existing_names is None:
        existing_names = set()

    for name in names:
        yield _unique_name(existing_names, name)
