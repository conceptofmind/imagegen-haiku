import haiku as hk

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, repeat, reduce
from .einops_exts import rearrange_many, repeat_many, check_shape

# helper functions

def exists(value):
    return value is not None

def identity(t, *args, **kwargs):
    return t

# norms and residuals

class LayerNorm(hk.Module):
    def __init__(Self, dim):
        super().__init__()
        self.gamma = hk.get_parameter()
        self.beta = hk.get_parameter()
    
    def call(self, x):
        return hk.LayerNorm()


class ChanLayerNorm(hk.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.g = hk.get_parameter("g", [1, dim, 1, 1], init=jnp.ones)

    def __call__(self, x):
        var = jnp.var(x, axis = 1, keepdims=True)
        mean = jnp.mean(x, axis = 1, keepdims=True)
        return (x - mean) / jnp.sqrt((var + self.epsilon)) * self.g

class Residual(hk.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# attention pooling

from functools import wraps, partial

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

class PercieverAttention(hk.Module):
    def __init__(self, *, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = hk.LayerNorm()
        self.norm_latents = hk.LayerNorm()

        self.to_q = hk.Dense(inner_dim, use_bias = False)
        self.to_kv = hk.Dense(inner_dim * 2, use_bias = False)

        self.to_out = hk.Sequential(
            hk.Dense(dim, use_bias = False),
            hk.LayerNorm(),
        )

    def __call__(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        kv_input = jnp.cat((x, latents), axis = -2)
        k, v = self.to_kv().split(2, axis = -1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(mask):
            max_neg_value = -jnp.finfo(sim.dtype).max

        attn = nn.softmax(sim, axis = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        out = self.to_out(out)

        return out

