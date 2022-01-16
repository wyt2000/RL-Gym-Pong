from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np
import cv2

class PB18111684(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)
        self.config = get_params_from_file('src.alg.PB18111684.rl_configs',params_name='params') # 传入参数
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

        self.frame_buffer_size = self.config['frame_buffer_size']
        self.frame_buffer = []
        self.old_action = 0

    def transform_image(self, image):
        '''
        Grayscale the input image:
        np.array(210, 160, 3) -> np.array(84, 84)
        '''
        assert image.shape == (210, 160, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (84, 110), interpolation=cv2.INTER_AREA)
        image = image[18:102, :]
        # cv2.imshow('show', image)
        # cv2.waitKey(0)
        return image

    def step(self, state):
        state = self.transform_image(state)
        
        if len(self.frame_buffer) == self.frame_buffer_size:
            action = self.explore(
                np.array(self.frame_buffer).astype(np.float32) / 255
            )
            self.old_action = action
            self.frame_buffer = []

        action = self.old_action
        self.frame_buffer.append(state)
        
        return action

    def explore(self, obs):
        assert obs.shape == (self.frame_buffer_size, 84, 84) and obs.dtype == np.float32
        action = self.ac_space.sample()
        return action

    def test(self):
        raise NotImplementedError
