from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np
import cv2
import collections
import torch

from src.alg.PB18111684.DQN import DQN

Experience = collections.namedtuple(
    'Experience', 
    field_names=['state', 'action', 'reward', 'new_state']
)

class PB18111684(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)
        self.config = get_params_from_file('src.alg.PB18111684.rl_configs',params_name='params') # 传入参数
        
        self.device = 'cpu'

        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

        self.frame_buffer_size = self.config['frame_buffer_size']
        self.frame_buffer = []
        self.old_action = 0
 
        self.state_shape = (self.frame_buffer_size, 84, 84)
        self.old_state = np.zeros(self.state_shape)
        self.reward = 0

        self.replay_buffer_size = self.config['replay_buffer_size']
        self.replay_buffer = collections.deque(
            maxlen = self.replay_buffer_size
        )

        self.epsilon = self.config['eps_init']
        self.eps_decay = self.config['eps_decay']
        self.eps_min = self.config['eps_min']
    
        self.Q = DQN(self.state_shape, self.action_dim)
        self.Q_target = DQN(self.state_shape, self.action_dim)

    def transform_image(self, image):
        '''
        Grayscale the input image:
        np.array(210, 160, 3) -> np.array(84, 84)
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (84, 110), interpolation=cv2.INTER_AREA)
        image = image[18:102, :]
        # cv2.imshow('show', image)
        # cv2.waitKey(0)
        return image

    def step(self, obs):
        obs = self.transform_image(obs)
        
        if len(self.frame_buffer) == self.frame_buffer_size:
            state = np.array(self.frame_buffer).astype(np.float32) / 255
            self.replay_buffer.append(
                Experience(
                    self.old_state,
                    self.old_action,
                    self.reward,
                    state
                )
            )
            action = self.explore(state)
            self.old_state = state
            self.old_action = action
            self.frame_buffer = []
            self.reward = 0

        action = self.old_action
        self.frame_buffer.append(obs)
        
        return action

    def explore(self, state):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        if np.random.random() < self.epsilon:
            action = self.ac_space.sample()
        else:
            state = np.array([state], copy=False)
            state = torch.tensor(state).to(self.device)
            _, action = torch.max(self.Q(state), dim=1)
            action = action.item()
        return action

    def get_reward(self, reward):
        self.reward += reward
        return

    def test(self):
        raise NotImplementedError
