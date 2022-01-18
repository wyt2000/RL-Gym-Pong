from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np
import cv2
import collections
import torch
import torch.nn as nn

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
        self.old_state = torch.zeros(self.state_shape)
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

        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.optimizer = torch.optim.Adam(
            self.Q.parameters(),
            lr = self.config['learning_rate']
        )

        self.sync_target_network = self.config['sync_target_network']
        self.state_count = 0

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
            state = torch.tensor(self.frame_buffer).to(torch.float32) / 255.0
            self.replay_buffer.append(
                Experience(
                    self.old_state.to(self.device),
                    torch.tensor(self.old_action).to(self.device),
                    torch.tensor(self.reward).to(self.device),
                    state.to(self.device)
                )
            )
            if len(self.replay_buffer) == self.replay_buffer_size:
                self.update()
            action = self.explore(state)
            self.old_state = state
            self.old_action = action
            self.frame_buffer = []
            self.reward = 0
            self.state_count += 1

        action = self.old_action
        self.frame_buffer.append(obs)
        
        return action

    def explore(self, state):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        if np.random.random() < self.epsilon:
            action = self.ac_space.sample()
        else:
            state = state.unsqueeze(0).to(self.device)
            _, action = torch.max(self.Q(state), dim=1)
            action = action.item()
        return action

    def update(self):
        indices = np.random.choice(
            len(self.replay_buffer),
            self.batch_size,
            replace = False
        )
        states = torch.stack([self.replay_buffer[i].state for i in indices])
        actions = torch.stack([self.replay_buffer[i].action for i in indices])
        rewards = torch.stack([self.replay_buffer[i].reward for i in indices])
        new_states = torch.stack([self.replay_buffer[i].new_state for i in indices])
        
        q_max = self.Q_target(new_states).max(dim=1)[0]
        q_max = q_max.detach()
        y = rewards + self.gamma * q_max
        q_values = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = nn.MSELoss()(q_values, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.state_count % self.sync_target_network == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_reward(self, reward):
        self.reward += reward
        return

    def show(self, image):
        cv2.imshow('show', image)
        cv2.waitKey(0)

    def test(self):
        raise NotImplementedError
