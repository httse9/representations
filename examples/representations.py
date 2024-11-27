import os

from absl import app
from absl import flags
import gin
import gym
from matplotlib import cm
from matplotlib import colors
import matplotlib.pylab as plt
import numpy as np
from os.path import join

from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import maxent_mdp_wrapper
from minigrid_basics.envs import maxent_mon_minigrid


FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'maxent_empty', 'Environment to run.')

flags.DEFINE_float('SR_step_size', 0.1, 'step size for SR.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')


flags.DEFINE_float('DR_step_size', 0.1, 'step size for DR.')
flags.DEFINE_float('MER_step_size', 0.1, 'step size for MER.')



flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')


def compute_SR(pi, P, gamma):
    """
    Compute the successor representation (SR).
    pi: policy with shape (S, A)  
    P: transition probability matrix with shape (S, A, S)
    gamma: discount factor, scalar
    """
    # compute P_pi, prob of state transition under pi
    P_pi = (P * pi[..., None]).sum(1)

    # compute SR
    n_states = P_pi.shape[0]
    SR = np.linalg.inv(np.eye(n_states) - gamma * P_pi)
    return SR

def compute_DR(pi, P, R):
    """
    Compute the default representation (DR).
    pi: default policy with shape (S, A)
    P: transition probability matrix with shape (S, A, S)
    R: reward vector with shape (S)
    """
    P_pi = (P * pi[..., None]).sum(1)
    DR = np.linalg.inv(np.diag(np.exp(-R)) - P_pi)
    return DR

def compute_MER(P, R, n_states, n_actions):
    """
    Compute the default representation (DR).
    P: transition probability matrix with shape (S, A, S)
    R: reward vector with shape (S)
    n_states: number of states, scalar
    n_actions: number of actions, scalar
    """
    pi_uniform = np.ones((n_states, n_actions)) / n_actions
    P_pi = (P * pi_uniform[..., None]).sum(1)
    MER = np.linalg.inv(np.diag(np.exp(-R)) / n_actions - P_pi)
    return MER

def main(argv):
    # print(argv)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    gin.parse_config_files_and_bindings(
        [os.path.join(maxent_mon_minigrid.GIN_FILES_PREFIX, '{}.gin'.format(FLAGS.env))],
        bindings=FLAGS.gin_bindings,
        skip_unknown=False)
    env_id = maxent_mon_minigrid.register_environment()

    env = gym.make(env_id)
    # Get tabular observation and drop the 'mission' field:
    env = maxent_mdp_wrapper.MDPWrapper(env)
    # env = coloring_wrapper.ColoringWrapper(env)


    R = env.rewards
    P = env.transition_probs


    n_states = env.num_states
    n_actions = env.num_actions

    print("Number of states:", n_states)
    print("Number of actions", n_actions)
    
    reward_grid = env.reward_grid
    plt.imshow(reward_grid.T)       # flip reward grid back
    plt.title("Reward Map")
    plt.show()

    # uniform random policy
    pi_uniform = np.ones((n_states, n_actions)) / n_actions

    start_state = 5

    # compute SR
    SR = compute_SR(pi_uniform, P, FLAGS.gamma)
    SR_start = SR[start_state].reshape(11, 11)
    # print(SR_start)
    plt.subplot(1, 3, 1)
    plt.imshow(SR_start)
    plt.title("SR")


    DR = compute_DR(pi_uniform, P, R)
    DR_start = DR[start_state].reshape(11, 11)
    # print(DR_start)
    plt.subplot(1, 3, 2)
    # plt.imshow(np.log(DR_start))
    plt.imshow(DR_start)
    plt.title("DR")

    MER = compute_MER(P, R, n_states, n_actions)
    MER_start = MER[start_state].reshape(11, 11)
    # print(MER_start)
    plt.subplot(1, 3, 3)
    # plt.imshow(np.log(MER_start))
    plt.imshow(MER_start)
    plt.title("MER")
    plt.show()

    def plot_top_eigenvector(rep, log=False):
        """
        rep: representation matrix (S, S)
        """
        lamb, e = np.linalg.eig(rep)
        idx = lamb.argsort()
        e = e.T[idx[::-1]]

        e0 = e[0].reshape(11, 11)
        # normalize to be unit vector
        e0 /= (e0 ** 2).sum()
        assert np.allclose((e0 ** 2).sum(), 1.0)

        # e0 *= (e0 < 0).astype(float) * -1
        if log:
            e0 = np.log(e0)

        # print(e0)
        # if log:
        #     # e_min = e0.min()
        #     # if e_min < 0:
        #     #     e0 -= e_min - 1e-5
        #     e0 = np.log(e0)
        plt.imshow(np.real(e0))

    plt.subplot(1, 3, 1)
    plot_top_eigenvector(SR)
    plt.title("SR")

    plt.subplot(1, 3, 2)
    plot_top_eigenvector(DR)
    plt.title("DR")

    plt.subplot(1, 3, 3)
    plot_top_eigenvector(MER, log=True)
    plt.title("MER")

    plt.show()
    # for rep in [DR, MER]:
    #     plot_top_eigenvector(rep, log=False)

    
    # s = env.reset()

    # print(env.state_to_pos)
    
    # for x in range(env.width):
    #     for y in range(env.height):
    #         # print(x,y )
    #         cell = env.grid.get(x, y)
    #         if cell is not None:
    #             if cell.type == "goal":
    #                 print("Goal position", x, y)

    # while True:
    #     s, _, done, _ = env.step(np.random.randint(4))
    #     if done:
    #         print(s['state'])
    #         print(env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width])
    #         print(env.state_to_pos[s['state']])
    #         break



#   s = env.reset()
#   print(s['state'])
#   print(env.agent_pos[0] + env.agent_pos[1] * env.width)
#   print(env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width])

#   o, r, done, d = env.step(1)
#   print(r, done, d)
#   print(o['state'])
#   print(env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width])


if __name__ == '__main__':
  app.run(main)