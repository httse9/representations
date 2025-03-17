import os

from absl import app
from absl import flags
import gin
import gym
from matplotlib import cm
from matplotlib import colors
import matplotlib.pylab as plt
import numpy as np
import pickle
from minigrid_basics.custom_wrappers import coloring_wrapper, mdp_wrapper
from gym_minigrid.wrappers import RGBImgObsWrapper
from minigrid_basics.envs import mon_minigrid

from minigrid_basics.examples.utility import *

FLAGS = flags.FLAGS

flags.DEFINE_integer('sr_pos', None,
                    'Successor state.')
flags.DEFINE_integer('n_pvf', None,
                    'nth pvf.')
flags.DEFINE_string('sr_image_file', None,
                    'Path prefix to use for saving the SR.')
flags.DEFINE_string('values_image_file', None,
                    'Path prefix to use for saving the observations.')
flags.DEFINE_string('env', 'classic_fourrooms', 'Environment to run.')
flags.DEFINE_float('tolerance', 0.0001, 'Error tolerance for value iteration.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor to use for SR.')
flags.DEFINE_float('gamma_options', 0.99, 'Discount factor to use for option policies.')
flags.DEFINE_float('step_size', 0.1, 'step size for SR.')
flags.DEFINE_float('step_size_options', 0.1, 'step size for option policies.')
flags.DEFINE_integer('n_iter', 4, 'Number of ROD Iterations')
flags.DEFINE_integer('n_steps', 100, 'Number of steps in each episode')
flags.DEFINE_float('p_option', 0.05, 'probability of sampling an option policy')
flags.DEFINE_integer('num_simulations', 100, 'number of times to run the ROD cycle')
flags.DEFINE_bool('show_graphs', True, 'show graphs')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override default parameter values '
    '(e.g. "MonMiniGridEnv.stochasticity=0.1").')

# def create_dir(dir):
#   if not os.path.exists(dir):
#     os.mkdir(dir)

# def graph_values(values, image_loc, env):
#   cmap = cm.get_cmap('plasma', 256)
#   norm = colors.Normalize(vmin=min(values), vmax=max(values))
#   obs_image = env.render_custom_observation(env.reset(), values, cmap)
#   m = cm.ScalarMappable(cmap=cmap, norm=norm)
#   m.set_array(obs_image)
#   plt.imshow(obs_image)
#   plt.colorbar(m)
#   plt.savefig(image_loc)
#   print('SAVED TO:', image_loc)
#   plt.clf()
#   plt.close()

# def get_eigenpurpose(e):
#   # r^e(s, s') = e(s') - e(s)
#   return np.tile(e, (e.shape[0], 1)) - np.tile(np.reshape(e, (-1, 1)), (1, e.shape[0]))

# def value_iteration(env, rewards):
  
#   values = np.zeros(env.num_states)
#   error = FLAGS.tolerance * 2
#   i = 0
#   while error > FLAGS.tolerance:
#     new_values = np.copy(values)
#     for s in range(env.num_states):
#       max_value = -np.inf
#       for a in range(env.num_actions):
#         curr_value = (np.dot(rewards[s, :], env.transition_probs[s, a, :]) +
#                 FLAGS.gamma * np.matmul(env.transition_probs[s, a, :],
#                                         values))
#         if curr_value > max_value:
#           max_value = curr_value
#       new_values[s] = max_value
#     error = np.max(abs(new_values - values))
#     values = new_values
#     i += 1
#     if i % 1000 == 0:
#       print('Error after {} iterations: {}'.format(i, error))
#   print('Found V* in {} iterations'.format(i))
#   return values 

# def qlearning(env, eigenpurpose, dataset):
#   q = np.zeros((env.num_states, env.num_actions))
#   for _ in range(1000):
#     for (s, a, sp) in dataset:
#       q[s, a] = q[s, a] + FLAGS.step_size_options * (eigenpurpose[s, sp] + FLAGS.gamma_options * np.max(q[sp, :]) - q[s, a])
#   return q

# def get_value(env, eigenpurpose, is_q=False, dataset=None):
#   if is_q:
#     return True, qlearning(env, eigenpurpose, dataset)
#   else:
#     return False, value_iteration(env, eigenpurpose)
  
# def generate_dataset(env, num_episodes, num_steps):
#   dataset = [] 
#   for _ in range(num_episodes):
#     env.reset()
#     s = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
#     for _ in range(num_steps):
#       a = np.random.choice(4)
#       env.step(a)
#       sp = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
#       dataset.append((s, a, sp))
#       s = sp
#     return dataset

# def get_SR(env, closed=True, dataset=None, prev_SR=None):
#   if closed:
#     P = env.transition_probs.sum(axis=1)*0.25
#     # Calculate successor representation
#     SR = np.linalg.inv(np.identity(n=P.shape[0], like=P) - FLAGS.gamma * P)  
#     return SR
#   else:
#     SR = np.zeros((env.num_states, env.num_states))
#     if prev_SR is not None:
#       SR = prev_SR
#     for _ in range(100):
#       for (s, a, sp) in dataset:
#         for i in range(env.num_states):
#           delta = (1 if s == i else 0) + (FLAGS.gamma * SR[sp, i]) - SR[s, i]
#           SR[s, i] = SR[s, i] + (FLAGS.step_size * delta)
#     return (SR + SR.T) / 2



def main(argv):
  np.random.seed(7)
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(mon_minigrid.GIN_FILES_PREFIX, '{}.gin'.format(FLAGS.env))],
      bindings=FLAGS.gin_bindings,
      skip_unknown=False)
  env_id = mon_minigrid.register_environment()
  env = gym.make(env_id)
  env = RGBImgObsWrapper(env)
  # Get tabular observation and drop the 'mission' field:
  env = mdp_wrapper.MDPWrapper(env)
  env = coloring_wrapper.ColoringWrapper(env)
  
  # import pdb; pdb.set_trace()
  timestep_list = []
  
  for simulation in range(FLAGS.num_simulations):
    
    print('-------------------------------------------')
    print(f'\nRun {simulation}\n')
    print('-------------------------------------------')

    dataset = []
    options = []
    SR = None
    state_visits = np.zeros(env.num_states)
    total_state_visits = np.zeros(env.num_states)
    state_image_dict = {}

    if FLAGS.show_graphs:
      create_dir(f'minigrid_basics/ROD/simulation_{simulation}')

    total_timesteps = 0   # keep track of time needed to visit all states

    i = 0
    
    while np.min(state_visits) < 1:
      obs = env.reset()
      s = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
      state_image_dict[s] = obs
      state_visits[s] += 1

      j = 1
      if np.min(state_visits) < 1:
        total_timesteps += 1

      # TODO: improve environment loop?
      # in class, instance variable keep tracking of current option

      while j < FLAGS.n_steps:
        if (np.random.random() > 1 - FLAGS.p_option) and (len(options) > 0):  # why not smaller than p_option...?
          option = np.random.choice(options)    # choose random option
          while option['termination'][s] == 0:    # termiantion condition? 0 for not terminate, 1 for terminate
            if j >= FLAGS.n_steps:    # break if episode ends
                break
            a = int(option['policy'][s])    # get action outputted by option policy
            obs, reward, done, _ = env.step(a)
            sp = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
            state_image_dict[sp] = obs
            state_visits[sp] += 1
            dataset.append((s, a, reward, sp))    # collect data
            s = sp 
            j = j + 1
            if np.min(state_visits) < 1:
              total_timesteps += 1
        else:
          a = np.random.choice(4)   # uniform exploration
          obs, reward, done, _ = env.step(a)
          sp = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
          state_image_dict[sp] = obs
          state_visits[sp] += 1
          dataset.append((s, a, reward, sp))
          s = sp
          j = j + 1
          if np.min(state_visits) < 1:
            total_timesteps += 1

      if FLAGS.show_graphs:
        env.render_state_visits(obs, state_visits, f'minigrid_basics/ROD/simulation_{simulation}/state_visits_{i}.png')

      print(f'Calculating SR {i}')
      # compute SR from collected dataset, start from previous SR iteration
      # how about terminal states? How do the cycle handle that?
      SR = get_SR(env, closed=False, dataset=dataset[-100:], prev_SR=SR, gamma=FLAGS.gamma, step_size=FLAGS.step_size)    
      
      #SR = get_SR(env)

      # for s in range(env.num_states):
      #   graph_values(SR[s], f'minigrid_basics/ROD/SR/SR_{s}.png', env)

      # Eigendecomposition of successor representation
      eigenvalues, eigenvectors = np.linalg.eig(SR)
      # eigenvalues = np.imag(eigenvalues)
      # eigenvectors = np.imag(eigenvectors)

      # U, s, V = np.linalg.svd(SR)
      # eigenvectors = V.T
      # eigenvalues = s
      
      idx = np.argsort(eigenvalues)[-1]
      
      if np.sum(eigenvectors[:, idx]) >= 0:   # -1 flip trick
        eigenvectors[:, idx] = eigenvectors[:, idx] * -1
      
      print(f'Generating option {i}')

      if FLAGS.show_graphs:
        graph_values(obs, eigenvectors[:,idx], f'minigrid_basics/ROD/simulation_{simulation}/eigenvector_{i}.png', env)

      # intrinsic reward
      r = get_eigenpurpose(eigenvectors[:,idx])

      ### compute eigenoption   --------------
      is_q, v = get_value(env, r, FLAGS.tolerance, FLAGS.gamma_options, FLAGS.step_size_options, True, dataset)
      option = {'instantiation': set(), 
                'policy': np.ones(env.num_states) * 5, 
                'termination': np.ones(env.num_states)}
      
      for s in range(env.num_states):
        if is_q:  # if learned Q wrt eigenpurpose
          if max(v[s, :]) > 0:
            option['instantiation'].add(s)
            option['policy'][s] = np.argmax(v[s, :])
            option['termination'][s] = 0
        else:   # if only learn state value
          max_val = 0
          for a in range(env.num_actions):
            q = np.dot(r[s, :], env.transition_probs[s, a, :]) + FLAGS.gamma * np.dot(env.transition_probs[s, a, :], v)
            if q > max_val:
              max_val = q
              option['instantiation'].add(s)
              option['policy'][s] = a
              option['termination'][s] = 0
      options.append(option)
      ### end compute eigenoption ------------------

      if FLAGS.show_graphs:
        env.render_option_policy(env.reset(), option, f'minigrid_basics/ROD/simulation_{simulation}/option_policy_{i}.png')

      i += 1
    if FLAGS.show_graphs:
      env.render_state_visits(env.reset(), (state_visits/np.sum(state_visits))*100, f'minigrid_basic/ROD/simulation_{simulation}/diffusion.png')

    total_state_visits = total_state_visits + state_visits    # total state visits across many ROD simulations
    
    timestep_list.append(total_timesteps)

    file = open(f'minigrid_basics/ROD/simulation_{simulation}/state_images.pkl', 'wb')
    pickle.dump(state_image_dict, file)
    file.close() 

  print(f'Statistics (n = {FLAGS.num_simulations}) on time to visit every state: ')
  print(f'Average: {np.mean(timestep_list)}')
  print(f'Median: {np.median(timestep_list)}')
  print(f'Standard Deviation: {np.std(timestep_list)}')
  print(f'Min: {np.min(timestep_list)}')
  print(f'Max: {np.max(timestep_list)}')

  if FLAGS.show_graphs:
      env.render_state_visits(env.reset(), (total_state_visits/np.sum(total_state_visits))*100, f'minigrid_basics/ROD/diffusion.png')

  env.close()


if __name__ == '__main__':
  app.run(main)
