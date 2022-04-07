# User Guide

`Module` is the main construct in JAX Metrics, it inherits from [treeo.Tree](https://github.com/cgarciae/treeo) and adds a couple of convenient methods. We recommend that you review the core concepts of Treeo but we will provide a brief overview. 

### Terminology
These are the core concepts from Treeo:

* **Type Annotation**: ([type hints](https://docs.python.org/3/library/typing.html)) types you set while defining a variable after the `:` symbol.
* **Field Declaration**: default values for class variables that are set using the `field` function.
* **Node Field**: A field that is declared as a node, that is, its content is part of the tree leaves.
* **Static Field**: A field that is declared as a static, that is, its content is not part of the leaves.
* **Field Kind**: An associated type, separate from the type annotation, that gives semantic meaning to the field.

In code these terms map to the following:

```python
class MyModule(jm.Module):
    #  field      annotation   ------------declaration---------------
    #    v            v        v                                    v
    some_field : jnp.ndarray = to.field(node=True, kind=jm.Parameter)
    #                                      ^                ^
    #                                 node status       field kind
```
Here if `node=False` it would mean that the field is static, else is a node. The previous is written more compactly as:

```python
class MyModule(jm.Module):
    some_field: jnp.ndarray = jm.Parameter.node()
```