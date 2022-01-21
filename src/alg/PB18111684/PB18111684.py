from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np
import cv2
import torch

from src.alg.PB18111684.DQN import DQN

class PB18111684(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)
        self.config = get_params_from_file('src.alg.PB18111684.rl_configs',params_name='params') # 传入参数

        self.device = 'cpu'
        self.frame_buffer_size = self.config['frame_buffer_size']
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

        self.frame_buffer = []
        self.old_action = 0
 
        self.state_shape = (self.frame_buffer_size, 84, 84)
    
        self.Q = DQN(self.state_shape, self.action_dim).to(self.device)
        self.load()

    def transform_image(self, image):
        '''
        Grayscale the input image:
        np.array(210, 160, 3) -> np.array(84, 84)
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (84, 110), interpolation=cv2.INTER_AREA)
        image = image[18:102, :]
        return image

    def step(self, obs):
        obs = self.transform_image(obs)

        if len(self.frame_buffer) == self.frame_buffer_size:
            state = torch.from_numpy(np.stack(self.frame_buffer)).to(torch.float32) / 255.0
            action = self.explore(state)
            self.old_action = action
            self.frame_buffer = []

        action = self.old_action
        self.frame_buffer.append(obs)
        
        return action

    def explore(self, state):
        state = state.unsqueeze(0).to(self.device)
        action = self.Q(state).max(dim=1)[1]
        action = action.item()
        return action

    def load(self):
        self.Q.load_state_dict(torch.load("src/alg/PB18111684/model_in.dat", map_location=self.device))

    def test(self):
        raise NotImplementedError
