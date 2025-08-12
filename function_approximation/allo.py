import jax
import jax.numpy as jnp

import optax
from flax import nnx

import cloudpickle
from functools import partial

import copy
import random

from rollout import compare_eigenvectors

jax.clear_caches()



    

class ALLO:
    
    
    def __init__(self, dummy_obs, args):

        self.args = args

        self.encoder = Encoder(obs_dim = dummy_obs.shape[-1], feat_dim = args.allo_feat_dim, eig_dim = args.eig_dim, 
                                duals_initial_val = args.duals_initial_val, barrier_initial_val = args.barrier_initial_val,
                                obs_type = args.obs_type, rngs = nnx.Rngs(0))
    
        if args.allo_step_size_decay:
                        
            when_to_start_decay = args.allo_gradient_steps * args.allo_step_size_delay
            
            self.allo_step_size_schedule = optax.linear_schedule(
                init_value = args.allo_step_size,
                end_value = args.allo_end_step_size,
                transition_steps = args.allo_gradient_steps - when_to_start_decay,
                transition_begin = when_to_start_decay
            )

        else:
            self.allo_step_size_schedule = args.allo_step_size
            
        self.encoder_optimizer = nnx.Optimizer(self.encoder, optax.adam(self.allo_step_size_schedule))
    
        
    def update_model(self, observations, next_observations, observations_2):
        aux = jnp.array([self.args.max_barrier_coefs, self.args.step_size_duals, self.args.min_duals, self.args.max_duals, self.args.barrier_scale])
        allo_loss = jitted_update_step(self.encoder, self.encoder_optimizer, observations, next_observations, observations_2, self.args.eig_dim, aux)
        return allo_loss
    
    def probe(self):        
        print(jnp.linalg.norm(jnp.asarray(self.encoder.duals)), jnp.linalg.norm(jnp.asarray(self.encoder.barrier_coefs)))


    def get_step_size(self):
        if self.args.allo_step_size_decay:
            # todo - (maybe) there is a neat-er way to get the learning rate
            return self.allo_step_size_schedule(self.encoder_optimizer.opt_state[0].count).item()
        else:
            return self.allo_step_size_schedule


    def checkpoint(self, ckpt_dir):

        model_graph, model_state = nnx.split(self.encoder) 
        
        data_to_save = {"model_graph": model_graph, "model_state": model_state}
                
        with open(ckpt_dir + "ALLO_graphNstate.pkl", "wb") as f:
            cloudpickle.dump(data_to_save, f)
            
    
    
@partial(nnx.jit, static_argnums=(5,))
def jitted_update_step(encoder, encoder_optimizer, observations, next_observations, observations_2, eig_dim, aux):

    max_barrier_coefs, step_size_duals, min_duals, max_duals, barrier_scale = aux
    
    def encoder_loss(encoder, observations, next_observations, observations_2):
        
        phi = encoder(observations)
        phi_2 = encoder(observations_2)

        next_phi = encoder(next_observations)

        graph_loss = 0.5 * ((phi - next_phi)**2).mean(0).sum()

        barrier_coefficients = encoder.barrier_coefs.clip(0, max_barrier_coefs)
        duals = encoder.duals #.clip(min_duals, max_duals)
                
        # Compute errors
        n = phi.shape[0]
        inner_product_matrix_1 = jnp.einsum(
            'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
        inner_product_matrix_2 = jnp.einsum(
            'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n

        error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(eig_dim))
        error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(eig_dim))

        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

        dual_loss_pos = (jax.lax.stop_gradient(jnp.asarray(duals)) * error_matrix).sum()

        dual_loss_neg = - step_size_duals * (duals * jax.lax.stop_gradient(error_matrix)).sum()

        quadratic_error_matrix = error_matrix_1 * error_matrix_2
        barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients[0,0]) * quadratic_error_matrix.sum()
        
        quadratic_error = jnp.clip(quadratic_error_matrix, 0, None).mean()
        barrier_loss_neg = -barrier_coefficients[0,0] * jax.lax.stop_gradient(quadratic_error)

        # Total loss
        allo_loss = dual_loss_pos + barrier_loss_pos + graph_loss + dual_loss_neg + (barrier_loss_neg * barrier_scale)
        
        indiv_losses = {
            'dual_loss_pos' : dual_loss_pos,
            'barrier_loss_pos' : barrier_loss_pos,
            'graph_loss' : graph_loss, 
            'dual_loss_neg' : dual_loss_neg,
            'barrier_loss_neg' : barrier_loss_neg,
            'allo_loss' : allo_loss
            }
        
        return allo_loss, indiv_losses

    (allo_loss, indiv_losses), grads = nnx.value_and_grad(encoder_loss, has_aux=True)(encoder, observations, next_observations, observations_2)
    
    encoder_optimizer.update(grads)
    
    return indiv_losses


def train_allo(all_obs, all_next_obs, evaluation_obs, true_eigvectors, ckpt_dir, args, wandb):
    
    random.seed(args.seed)
    
    print("Training ALLO....")

    allo_model = ALLO(all_obs[0], args)
    
    tot_samples = len(all_obs)

    all_indices = [i for i in range(tot_samples)]
    b2_indices = copy.deepcopy(all_indices)

    loss_names = ['dual_loss_pos', 'barrier_loss_pos', 'graph_loss', 'dual_loss_neg', 'barrier_loss_neg', 'allo_loss']
    
    gradient_step = 0

    random.shuffle(all_indices)
    random.shuffle(b2_indices)
    
    start_idx = 0
    
        
    while gradient_step < args.allo_gradient_steps:
        
        end_idx = min(start_idx + args.allo_batch_size, tot_samples)

        obs = all_obs[all_indices[start_idx : end_idx]]
        next_obs = all_next_obs[all_indices[start_idx : end_idx]]
        
        obs2 = all_obs[b2_indices[start_idx : end_idx]]
            
        losses = allo_model.update_model(observations = jnp.array(obs),
                                        next_observations = jnp.array(next_obs), 
                                        observations_2 = jnp.array(obs2))
        
        gradient_step += 1
        start_idx += args.allo_batch_size
        
        if end_idx == tot_samples:
            random.shuffle(all_indices)
            random.shuffle(b2_indices)
            start_idx = 0

        for ln in loss_names:
            wandb.log({f"ALLO_losses/{ln}": losses[ln].item()})
        
        wandb.log({'Step_Sizes/ALLO': allo_model.get_step_size()})

        if gradient_step % 100 == 0:
            allo_model.checkpoint(ckpt_dir)
            
            allo_eigenvectors = allo_model.encoder(evaluation_obs)
            allo_eigenvectors = jnp.array(allo_eigenvectors[:, 1:])  
            similarities = compare_eigenvectors(allo_eigenvectors, true_eigvectors)
            
            for i, sim in enumerate(similarities):
                wandb.log({f"EigenVector_Similarity/{i}": sim})
            
            avg_sim = sum(similarities) / len(similarities)
            
            wandb.log({"EigenVector_Similarity/avg": avg_sim})
            wandb.log({"ALLO_Losses/barrier_coeff": allo_model.encoder.barrier_coefs[0,0].item()})
            wandb.log({"ALLO_Losses/dual_norm": jnp.linalg.norm(jnp.asarray(allo_model.encoder.duals)).item()})

            print(f"ALLO - {gradient_step: >6} / {args.allo_gradient_steps} : Avg Sim: {avg_sim:>4.3f}")


    allo_model.checkpoint(ckpt_dir)
    
    print(f"Finished training ALLO. Model saved at {ckpt_dir}")
    print()
    
    return allo_model
