import numpy as np
import os
from matplotlib import cm
from matplotlib import colors
import matplotlib.pylab as plt
import scipy.stats

def create_dir(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)

def graph_values(obs, values, image_loc, env):
  cmap = cm.get_cmap('plasma', 256)
  norm = colors.Normalize(vmin=min(values), vmax=max(values))
  obs_image = env.render_custom_observation(obs, values, cmap)
  m = cm.ScalarMappable(cmap=cmap, norm=norm)
  m.set_array(obs_image)
  plt.figure()
  plt.imshow(obs_image)
  plt.colorbar(m)
  plt.savefig(image_loc)
  print('SAVED TO:', image_loc)
  plt.clf()
  plt.close()

def get_eigenpurpose(e):
  # r^e(s, s') = e(s') - e(s)
  return np.tile(e, (e.shape[0], 1)) - np.tile(np.reshape(e, (-1, 1)), (1, e.shape[0]))

def value_iteration(env, rewards, tolerance, gamma):
  
  values = np.zeros(env.num_states)
  error = tolerance * 2
  i = 0
  while error > tolerance:
    new_values = np.copy(values)
    for s in range(env.num_states):
      max_value = -np.inf
      for a in range(env.num_actions):
        curr_value = (np.dot(rewards[s, :], env.transition_probs[s, a, :]) +
                gamma * np.matmul(env.transition_probs[s, a, :],
                                        values))
        if curr_value > max_value:
          max_value = curr_value
      new_values[s] = max_value
    error = np.max(abs(new_values - values))
    values = new_values
    i += 1
  return values 

def qlearning(env, eigenpurpose, dataset, step_size, gamma):
  q = np.zeros((env.num_states, env.num_actions))
  for _ in range(1000):
    for (s, a, r, sp) in dataset:
      q[s, a] = q[s, a] + step_size * (eigenpurpose[s, sp] + gamma * np.max(q[sp, :]) - q[s, a])
  return q

def get_value(env, eigenpurpose, tolerance, gamma, step_size, is_q=False, dataset=None):
  if is_q:
    return True, qlearning(env, eigenpurpose, dataset, step_size, gamma)
  else:
    return False, value_iteration(env, eigenpurpose, tolerance, gamma)
  
def get_bottleneck_value(env, dataset, step_size, gamma, acquired_state, acquired_action):
  q = np.zeros((env.num_states, env.num_actions))
  for _ in range(1000):
    for (s, a, r, sp) in dataset:
      bottleneck_reward = 1 if (sp == acquired_state and a == acquired_action) else 0
      q[s, a] = q[s, a] + step_size * (bottleneck_reward + gamma * np.max(q[sp, :]) - q[s, a])
      if bottleneck_reward == 1 or sp == acquired_state:
        break
  
  return q

def generate_dataset(env, num_episodes, num_steps):
  dataset = [] 
  for _ in range(num_episodes):
    env.reset()
    s = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
    for _ in range(num_steps):
      a = np.random.choice(4)
      env.step(a)
      sp = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
      dataset.append((s, a, sp))
      s = sp
    return dataset

def get_SR(env, gamma, step_size, closed=True, dataset=None, prev_SR=None):
  if closed:
    P = env.transition_probs.sum(axis=1)*0.25
    # Calculate successor representation
    SR = np.linalg.inv(np.identity(n=P.shape[0], like=P) - gamma * P)  
    return SR
  else:
    SR = np.zeros((env.num_states, env.num_states))
    if prev_SR is not None:
      SR = prev_SR
    for _ in range(100):
      for (s, a, r, sp) in dataset:
        for i in range(env.num_states):
          delta = (1 if s == i else 0) + (gamma * SR[sp, i]) - SR[s, i]
          SR[s, i] = SR[s, i] + (step_size * delta)
    return (SR + SR.T) / 2
    #return SR

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def update_option_value(rewards, q, option_idx, s, sp, gamma, step_size):
  cumulative_reward = 0
  for r in reversed(rewards):
    cumulative_reward = r + gamma * cumulative_reward
  q[s, option_idx] = q[s, option_idx] + step_size * (cumulative_reward + (gamma**len(rewards)) * np.max(q[sp, :]) - q[s, option_idx])

def take_option(option, q, goal_reached, starting_s, env, gamma, step_size):
  rewards = []
  s = starting_s
  sp = starting_s
  while option['termination'][s] == 0 and not goal_reached:
    a = int(option['policy'][s])
    _, reward, done, _ = env.step(a)
    rewards.append(reward)
    sp = env.pos_to_state[env.agent_pos[0] + env.agent_pos[1] * env.width]
    q[s, a] = q[s, a] + step_size * (reward + gamma * np.max(q[sp, :]) - q[s, a])
    s = sp
    if done:
      goal_reached = True
      break
  
  return rewards, goal_reached, sp