from minigrid_basics.examples.ROD_cycle import RODCycle
import os
import numpy as np
from flint import arb_mat, ctx
from itertools import islice

ctx.dps = 100   # important

# testing imports
import gym
import gin
from minigrid_basics.reward_envs import maxent_mon_minigrid
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import subprocess
import glob
import pickle

def power_iteration(A, num_iters=1000, tol=1e-9):
    b = np.random.rand(A.shape[1])
    b = b / np.linalg.norm(b)

    for _ in range(num_iters):
        b_new = A @ b
        b_new = b_new / np.linalg.norm(b_new)
        if np.linalg.norm(b_new - b) < tol:
            break
        b = b_new

    return b

class RODCycle_DR(RODCycle):

    def __init__(self, env, n_steps=100, p_option=0.05, dataset_size=None, learn_rep_iteration=10, representation_step_size=0.1,
                 gamma=0.99, num_options=None, eigenoption_step_size=0.1, lambd=1.3, plot=True):
        super().__init__(env, n_steps=n_steps, p_option=p_option, dataset_size=dataset_size, learn_rep_iteration=learn_rep_iteration,
            representation_step_size=representation_step_size, gamma=gamma, num_options=num_options, eigenoption_step_size=eigenoption_step_size,
            plot=plot)
        
        self.lambd = lambd # lambda for the DR
        self.plot_path = "minigrid_basics/DR_ROD"
        
        self.reset()
        

    def reset(self):
        """
        Initialize DR TD learning
        """
        super().reset()
        self.representation = np.eye(self.env.num_states)


    def learn_representation(self):
        """
        DR TD learning
        """
        if self.dataset_size is not None:
            dataset = self.dataset[-self.dataset_size:]
        else:
            dataset = self.dataset

        # do one backward pass through dataset for theoretical guarantee
        for (s, a, r, ns) in reversed(dataset):
            indicator = np.zeros((self.env.num_states))
            indicator[s] = 1
            self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])

        # remaining iterations, do forward pass
        for _ in range(self.learn_rep_iteration - 1):
            for (s, a, r, ns) in dataset:

                indicator = np.zeros((self.env.num_states))
                indicator[s] = 1

                self.representation[s] += self.representation_step_size * (np.exp(r / self.lambd) * (indicator + self.representation[ns]) - self.representation[s])


    def compute_eigenvector(self):
        """
        NEW WAY
        """
        DR = (self.representation + self.representation.T) / 2

        # get idx of visited states
        visited_idx = (DR.sum(1) != 1)
        DR_visited = DR[visited_idx][:, visited_idx]

        # do eigendecomposition only on visited states for stability
        DR_visited = arb_mat(DR_visited.tolist())
        lamb_visited, e_visited = DR_visited.eig(right=True, algorithm="approx", )
        lamb_visited = np.array(lamb_visited).astype(np.clongdouble).real.flatten()
        e_visited = np.array(e_visited.tolist()).astype(np.clongdouble).real.astype(np.float32)

        # sort eigenvalue and eigenvectors
        idx = np.argsort(lamb_visited)
        lamb_visited = lamb_visited[idx]
        e_visited = e_visited.T[idx]

        # get top eigenvector, assert same sign for all entries
        e0_visited = e_visited[-1]

        # debug
        # if not ((e0_visited <= 0).all() or (e0_visited >= 0).all()):
        #     job_id = os.environ.get('SLURM_JOB_ID')
        #     with open(f"minigrid_basics/fail-{job_id}.pkl", "wb") as f:
        #         pickle.dump(self.representation, f)


        # multiplicity of top eigenvalue might be greated than 1
        # positive top eigenvector hidden in space spanned by the
        # corresponding eigenvectors
        # if this happens, use power iteration

        if not ((e0_visited < 0).all() or (e0_visited > 0).all()):
            DR_visited = DR[visited_idx][:, visited_idx]
            e0_visited = power_iteration(DR_visited)

        assert (e0_visited < 0).all() or (e0_visited > 0).all()

        # project back to full state space
        e0 = np.zeros_like(visited_idx).astype(float)
        e0[visited_idx] = e0_visited

        # take log
        if e0.sum() < 0:
            e0 *= -1
        log_e0 = np.where(e0 > 0, np.log(e0), e0)       # apply log only on positive entries

        # normalize
        if (log_e0 != 0).any():
            log_e0 /= np.sqrt(log_e0 @ log_e0)
        else:
            log_e0 += 1 / np.sqrt(self.env.num_states)
        assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0



    def compute_eigenvector_old(self):
        """
        *** OLD
        *** Found better way of doing it
        DR eigenvector. Take log
        Return loged eigenvector
        """
        DR = (self.representation + self.representation.T) / 2
        num_unvisited_states = (DR.sum(1) == 1.).astype(int).sum()

        DR = arb_mat(DR.tolist())
        lamb, e = DR.eig(right=True, algorithm="approx", )
        lamb = np.array(lamb).astype(np.clongdouble).real.flatten()
        e = np.array(e.tolist()).astype(np.clongdouble).real.astype(np.float32)

        idx = np.argsort(lamb)
        lamb = lamb[idx]
        e = e.T[idx]

        # tentative top eigenvector
        e0 = e.T[-num_unvisited_states-1]       ## !!! bug

        # handle edge case
        if not ((e0 <= 0).all() or (e0 >= 0).all()):
            print("edge")
            # search from eigenvalues 1
            for v in reversed(e):
                if ((v <= 0).all() or (v >= 0).all()) and (v != 0).astype(int).sum() > 1:
                    e0 = v
                    break

        # assert entries are positive before taking log
        # Note: there may exist 0 entries due to unvisited states
        if e0.sum() < 0:
            e0 *= -1

        # code for debug
        # if not (e0 >= 0).all():
        #     job_id = os.environ.get('SLURM_JOB_ID')
        #     with open(f"minigrid_basics/fail-{job_id}.pkl", "wb") as f:
        #         pickle.dump(self.representation, f)

        assert (e0 >= 0).all()

        log_e0 = np.where(e0 > 0, np.log(e0), e0)       # apply log only on positive entries

        # normalize
        if (log_e0 != 0).any():
            log_e0 /= np.sqrt(log_e0 @ log_e0)
        else:
            print("ZERO")
            log_e0 += 1 / np.sqrt(self.env.num_states)
        assert np.isclose(log_e0 @ log_e0, 1.0)

        return log_e0
    


"""
If we init D to be zeros, then more visited states have higher values. 
Eigenvector assign higher values to more visited states
Flip everything to be positive, take eigendecomposition.
We have very negative for less visited states.
Flip this eigenvector, we get to visit states that are least visited.

However, states that have low reward have similar effect. 
They have small values, and have very negative eigenvector entries after logging.
After flipping, algorithm encourages to go to them. 


"""



    

if __name__ == "__main__":
    

    env_name = "gridroom_25"

    gin.parse_config_file(os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, f"{env_name}.gin"))
    env_id = maxent_mon_minigrid.register_environment()

    np.random.seed(0)

    env = gym.make(env_id, seed=42, no_goal=True)
    env = maxent_mdp_wrapper.MDPWrapper(env, )

    rodc = RODCycle_DR(env, learn_rep_iteration=1, num_options=1, representation_step_size=0.03, dataset_size=100, p_option=0.1, n_steps=200)
    
    # trying to find settings where it fails, and the I can use my fix to fix it.
    # rodc = RODCycle_DR(env, learn_rep_iteration=1, num_options=1, representation_step_size=0.01, dataset_size=100)

    rewards, visit_percentage = rodc.rod_cycle(n_iterations=100)

    for i in range(1, len(rewards)):
        rewards[i] += rewards[i - 1]

    # save video
    os.chdir("minigrid_basics/DR_ROD")
    # for prefix in ['option', 'cumulative_visit', 'eigenvector']:
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'iteration%d.png', '-r', '30','-pix_fmt', 'yuv420p', 
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-y', f'ROD_{env_name}.mp4'
    ])

    for file_name in  glob.glob("*.png"):
        os.remove(file_name)
