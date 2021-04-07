import functools
from typing import Any, Callable
from flax import optim
from flax.training import common_utils
import jax
from jax import random
import jax.numpy as jnp
from functools import partial
import sys
from flax import linen as nn
import numpy as np
import os

from flax.training import checkpoints

ModuleDef = Any

# define model - for the inequality to exist needs to have 4 conv + batchnorm
# one of the conv layers needs to be logged and some kind of residual structure
class ResNet50(nn.Module):
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(
            features=self.num_filters * 4,
            kernel_size=(1, 1),
            name="b1_Conv_2",
            use_bias=False,
            dtype=self.dtype,
        )(x)
        residual = nn.Conv(
            features=self.num_filters * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            name="b1_conv_proj",
            use_bias=False,
            dtype=self.dtype,
        )(residual)
        residual = nn.BatchNorm(
            name="b1_norm_proj",
            use_running_average=False,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )(residual)
        x = residual + y
        residual = x
        y = nn.Conv(
            features=self.num_filters * 4,
            kernel_size=(3, 3),
            strides=(1, 1),
            name="b2_Conv_1",
            use_bias=False,
            dtype=self.dtype,
        )(x)
        var_tracker = self.variable(
            "zzz_grad_stats", "b2_conv_proj_dummy", jnp.zeros, (residual.shape)
        )
        residual = residual + var_tracker.value
        residual = nn.Conv(
            features=self.num_filters * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            name="b2_conv_proj",
            use_bias=False,
            dtype=self.dtype,
        )(residual)
        x = residual + y
        return x


# initialize model and get parameters and intial state
def initialized(key, model):
    input_shape1 = (1, 10, 10, 1)

    @functools.partial(jax.jit)
    def init(params, x):
        return model.init(params, x)

    variables = init(
        {"params": key},
        jnp.ones(input_shape1, model.dtype),
    )
    model_state, params = variables.pop("params")
    return params, model_state


def train_step_N(apply_fn, optimizer, state_model_state, batch):
    """Perform a single training step."""

    def loss_fn(params, dummy_variable):
        """loss function used for training."""
        variables = {
            "params": params,
            "zzz_grad_stats": dummy_variable,
            "batch_stats": state_model_state["batch_stats"],
        }
        # fwd pass
        logits, new_model_state = apply_fn(
            variables,
            batch["image"],
            mutable=["batch_stats"],
        )
        # compute loss value
        loss = -jnp.sum(batch["label"] * logits)
        return loss, (new_model_state, logits)

    # get gradient compute function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=[0, 1])
    aux, grad = grad_fn(optimizer, state_model_state["zzz_grad_stats"])

    # return grad and don't return grad_stats (!)
    return grad[0], None


def train_step_G(apply_fn, optimizer, state_model_state, batch):
    """Perform a single training step."""

    def loss_fn(params, dummy_variable):
        """loss function used for training."""
        variables = {
            "params": params,
            "zzz_grad_stats": dummy_variable,
            "batch_stats": state_model_state["batch_stats"],
        }
        # fwd pass
        logits, new_model_state = apply_fn(
            variables,
            batch["image"],
            mutable=["batch_stats"],
        )
        # compute loss value
        loss = -jnp.sum(batch["label"] * logits)
        return loss, (new_model_state, logits)

    # get gradient compute function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=[0, 1])
    aux, grad = grad_fn(optimizer, state_model_state["zzz_grad_stats"])

    # return grad and grad_stats (!)
    return grad[0], grad[1]


# auxiliary function to get number of NaN in pytree
number_of_nans = lambda y_tree: jnp.array(
    jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: jnp.isnan(x).sum(), y_tree)
    )
).sum()


def train_and_evaluate():
    # XLA flags to avoid arithemetic modifying flags
    os.environ[
        "XLA_FLAGS"
    ] = "--xla_cpu_enable_fast_math=false --xla_cpu_fast_math_honor_nans=true --xla_cpu_fast_math_honor_infs=true --xla_cpu_fast_math_honor_division=true --xla_cpu_fast_math_honor_functions=true --xla_cpu_enable_fast_min_max=false --xla_gpu_enable_fast_min_max=false --xla_allow_excess_precision=false"

    # set 0 as random seed
    rng = random.PRNGKey(0)

    # create model with oryx core passed into the bwd to collect gradients
    model = ResNet50(dtype=jnp.float32)

    # get parameters of model
    params, model_state_N = initialized(rng, model)
    # in case you want to make sure that there are no zeros in params - not necessary so far
    # params = jax.tree_util.tree_map(lambda x: jnp.where(x == 0, .1, x * .001), params)

    # apply static parameters to train_step and jit the resulting function
    p_train_step_N = jax.jit(
        functools.partial(
            train_step_N,
            model.apply,
        )
    )

    # apply static parameters to train_step and jit the resulting function
    p_train_step_G = jax.jit(
        functools.partial(
            train_step_G,
            model.apply,
        )
    )

    batch_size = 6  # smallest batch size possible - 5 no error
    image_dim = 10  # smallest x and y dimension, however I have not tested x!=y combinations

    batch = {
        "image": jnp.arange(batch_size * image_dim * image_dim * 1).reshape(
            (batch_size, image_dim, image_dim, 1)
        ),
        "label": jnp.arange(batch_size * image_dim * image_dim * 256).reshape(
            (batch_size, image_dim, image_dim, 256)
        ),
    }

    # perform 1xfwd and 1xbwd to get gradients an logged intermediate gradients
    grad_N, grad_stats_N = p_train_step_N(params, model_state_N, batch)
    grad_G, grad_stats_G = p_train_step_G(params, model_state_N, batch)

    # check for NaNs in grads and logged stats
    print(
        "nan count:"
        + str(number_of_nans(grad_N))
        + " "
        + str(number_of_nans(grad_stats_N))
        + " "
        + str(number_of_nans(grad_G))
        + " "
        + str(number_of_nans(grad_stats_G))
    )

    # check for difference in both grads
    devs = jnp.sum(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_multimap(
                    lambda x, y: (x != y).sum(),
                    grad_N,
                    grad_G,
                )
            )
        )
    )

    # raise exception when different grads
    if devs != 0:
        raise Exception("states unequal: " + str(devs))

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return None


if __name__ == "__main__":
    train_and_evaluate()

