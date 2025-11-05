import jax
from jax import jit, lax
import jax.numpy as jnp
from flax import nnx
from functools import partial


jax.config.update("jax_enable_x64", True)


@partial(nnx.jit, static_argnums=())
def log_objective_step(encoder, obs, rewards, next_obs, terminals, obs2, optimizer,):
    """
    Minimizes the log objective: log [u^T (R-P) u]


    encoder: network for outputting representation
    obs: current obs/state
    rewards: reward of current obs/state
    next obs: next obs/state
    terminals: whether current obs/state is a terminal state
    obs2: another set of obs/state for computing quadratic penalty term in allo
    optimizer: optimizer
    test_set: a set of all states of the environment for evaluating the learned represnetations
    true_eigvec: the true eigvec of the DR
    """

    def loss_fn(encoder, obs, rewards, next_obs, terminals, obs2):

        barrier_coefficients = encoder.barrier_coefs

        # To ensure that the network outputs positive values, we treat the network output
        # as the log of the DR. So take exponential here to get eigenvector
        log_phi = encoder(obs)
        log_next_phi = encoder(next_obs)
        log_phi_2 = encoder(obs2)

        phi = jnp.exp(log_phi)
        next_phi = jnp.exp(log_next_phi)
        phi_2 = jnp.exp(log_phi_2)

        rewards = jnp.expand_dims(rewards, 1)

        graph_loss = 2 * phi + jnp.log(1 - jnp.exp(log_next_phi + rewards - log_phi))
        graph_loss = graph_loss.mean()

        quadratic_loss = (jnp.exp(2 * log_phi) - 1) * (jnp.exp(2 * log_phi_2) - 1)
        quadratic_loss = quadratic_loss.mean()

        loss = graph_loss + barrier_coefficients * quadratic_loss
        


        # individual_losses = lax.stop_gradient(dict(
            
        # ))

        return loss#, individual_losses

    loss, grads = nnx.value_and_grad(loss_fn, has_aux=False)(encoder, obs, rewards, next_obs, terminals, obs2)
    optimizer.update(grads)

    return loss