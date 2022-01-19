from gym.spaces import Discrete
import gym
import gym.envs

import numpy as np
import cv2
import collections
import torch
import torch.nn as nn
import tqdm
import json

Experience = collections.namedtuple(
    'Experience', 
    field_names=['state', 'action', 'reward', 'new_state']
)

class DQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_shape[0],
                out_channels = 32,
                kernel_size  = 8,
                stride       = 4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 32,
                out_channels = 64,
                kernel_size  = 4,
                stride       = 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 64,
                out_channels = 64,
                kernel_size  = 3,
                stride       = 1
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self.conv(
                torch.zeros(1, *in_shape)
        ).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(
                in_features  = conv_out_size,
                out_features = 512
            ),
            nn.ReLU(),
            nn.Linear(
               in_features   = 512,
               out_features  = n_actions 
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class PB18111684():
    def __init__(self,ob_space,ac_space):
        assert isinstance(ac_space, Discrete)

        self.device = 'cpu'
        self.frame_buffer_size = 4
        self.replay_buffer_size = 10000
        self.epsilon = 1.0
        self.eps_decay = 0.999985
        self.eps_min = 0.02
        self.batch_size = 32
        self.gamma = 0.99
        self.sync_target_network = 1000
        self.learning_rate = 1e-4
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

        self.frame_buffer = []
        self.old_action = 0
 
        self.state_shape = (self.frame_buffer_size, 84, 84)
        self.old_state = torch.zeros(self.state_shape)
        self.reward = 0

        self.replay_buffer = collections.deque(
            maxlen = self.replay_buffer_size
        )

    
        self.Q = DQN(self.state_shape, self.action_dim).to(self.device)
        self.Q_target = DQN(self.state_shape, self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.Q.parameters(),
            lr = self.learning_rate
        )

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
            state = torch.from_numpy(np.stack(self.frame_buffer)).to(torch.float32) / 255.0
            self.replay_buffer.append(
                Experience(
                    self.old_state,
                    torch.tensor(self.old_action),
                    torch.tensor(self.reward),
                    state
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
            action = self.Q(state).max(dim=1)[1]
            action = action.item()
        return action

    def update(self):
        indices = np.random.choice(
            len(self.replay_buffer),
            self.batch_size,
            replace = False
        )
        states = torch.stack([self.replay_buffer[i].state for i in indices]).to(self.device)
        actions = torch.stack([self.replay_buffer[i].action for i in indices]).to(self.device)
        rewards = torch.stack([self.replay_buffer[i].reward for i in indices]).to(self.device)
        new_states = torch.stack([self.replay_buffer[i].new_state for i in indices]).to(self.device)
        
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

    def save(self):
        torch.save(self.Q.state_dict(), "model.dat")

    def load(self):
        self.Q.load_state_dict(torch.load("model_in.dat", map_location=self.device))

    def test(self):
        raise NotImplementedError
        
def train(env, policy, num_train_episodes, is_render):
    try:
        #policy.load()
        log = []
        with tqdm.tqdm(total=num_train_episodes, ncols=100) as pbar:
            max_avg_reward = -20
            avg_reward = 0
            i = 0
            for j in range(num_train_episodes):
                obs = env.reset()
                done = False
                ep_ret = 0
                ep_len = 0
                while not(done):
                    # if is_render:
                    #     env.render()
                    ac = policy.step(obs)
                    obs, reward, done, _ = env.step(ac)
                    policy.get_reward(reward)
                    ep_ret += reward
                    ep_len += 1
                avg_reward = (avg_reward * i + ep_ret) / (i + 1)
                i += 1
                if i == 100:
                    log.append({'Episode':i,'reward':avg_reward})
                    if avg_reward > max_avg_reward:
                        max_avg_reward = avg_reward
                        policy.save()
                    avg_reward = 0
                    i = 0
                pbar.set_description(f"Epoch: {j}, avg_reward: {avg_reward:.2f}")
                pbar.update(1)
    finally:
        with open('reward.log', 'w') as f:
            dump = json.dumps(
                log,
                indent=4,
                separators=(',', ': ')
            )
            f.write(dump)

            
env = gym.make('Pong-v0')
policy = PB18111684(env.observation_space, env.action_space)
train(env, policy, 1000, False)
