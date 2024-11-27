# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).

This extends the TabularWrapper, which converts a MiniGridEnv to a tabular MDP.

This iterates through the states (assuming the number of states is given by
width * height) and constructs the transition probabilities. It assumes, like
in the stochasticity setting of our local copy of MiniGridEnv, that the
stochasticity probability mass is distributed evenly across the three actions
different from the one chosen, and staying in-place.
"""

import numpy as np
from minigrid_basics.custom_wrappers import tabular_wrapper


def get_next_state(pos, env):
  """Return the next state.

  Args:
    pos: pair of ints, current agent position.
    env: MiniGrid environment.

  Returns:
    The next agent position.
  """
  # get next state
  next_pos = pos
  fwd_pos = env.front_pos
  cell = env.grid.get(*fwd_pos)
  if cell is None or cell.can_overlap():
    next_pos = fwd_pos

  return next_pos


class MDPWrapper(tabular_wrapper.TabularWrapper):
  """Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).
  """

  def __init__(self, env, tile_size=8, get_rgb=False, goal_absorbing=False):
    """
    goal_absorbing: whether the goal states are absorbing. If True, goal state
      transition to itself with prob 1. If False, goal state transition to 
      the absorbing state with prob 1, and the entries in the transition prob
      matrix corresponding to the goal states are all 0.
    """
    super().__init__(env, tile_size=tile_size, get_rgb=get_rgb)

    assert self.stochasticity == 0

    self.num_actions = len(env.actions)
    self.transition_probs = np.zeros((self.num_states, self.num_actions,
                                      self.num_states))
    self.rewards = np.zeros((self.num_states, ))    # reward only depends on state
    env = self.unwrapped

    # state numbers are ordered starting from 1st row (left to right), 
    # then 2nd row, then ...
    for y in range(self.height):
      for x in range(self.width):

        s1 = self.pos_to_state[x + y * env.width]

        if s1 < 0:  # Invalid position. (ignore walls)
          continue

        cell = self.grid.get(x, y)

        # take care of goal states
        if cell is not None and cell.type == 'goal':
          # ignore transition, cause transition prob from goal is all zeros
          # set reward of goal states
          self.rewards[s1] = self.reward_dict[env._raw_grid[x, y]] 

          if goal_absorbing:
            # if treat goal as absorbing state
            for a in range(self.num_actions):
              self.transition_probs[s1, a, s1] = 1.

            # make sure reward at goal state is 0 if absorbing
            assert self.rewards[s1] == 0.

          continue

        # non-goal states
        env.agent_pos = np.array([x, y])
        r = env._reward() # pylint: disable=protected-access
        for a in range(self.num_actions):
          env.agent_dir = a
          next_pos = get_next_state([x, y], env)
          s2 = self.pos_to_state[next_pos[0] + next_pos[1] * env.width]
          self.transition_probs[s1, a, s2] = 1.
          self.rewards[s1] = r

    self.nonterminal_idx = (self.transition_probs.sum(-1).sum(-1) != 0)
