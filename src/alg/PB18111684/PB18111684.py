from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np

class PB18111684(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)
        self.config = get_params_from_file('src.alg.PB18111684.rl_configs',params_name='params') # 传入参数
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

    def step(self, state):
        # Choose an action randomly.
        # self.explore(state)
        action = self.ac_space.sample()
        return action

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError