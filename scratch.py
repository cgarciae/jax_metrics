import jax
import jax.numpy as jnp


# create your pairwise function
def distance(a, b):
    return jnp.linalg.norm(a - b)


# vmap based combinator to operate on all pairs
def all_pairs(f):
    f = jax.vmap(f, in_axes=(None, 0))
    f = jax.vmap(f, in_axes=(0, None))
    return f


# transform to operate over sets
distances = all_pairs(distance)

# create some test data
A = jnp.array([[0, 0], [1, 1], [2, 2]])
B = jnp.array([[-10, -10], [-20, -20]])

# compute distance of the first two points
d00 = distance(A[0], B[0])
# 14.142136

# now compute the distance of all pairs
D = distances(A, B)
# [[14.142136 28.284271]
#  [15.556349 29.698484]
#  [16.970562 31.112698]]
