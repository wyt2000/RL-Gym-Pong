import os
from time import sleep
import gym
import tqdm

def train(env, policy, num_train_episodes, is_render):
    with tqdm.tqdm(total=num_train_episodes, ncols=100) as pbar:
        avg_reward = 0
        for j in range(num_train_episodes):
            obs = env.reset()
            done = False
            ep_ret = 0
            ep_len = 0
            while not(done):
                if is_render:
                    env.render()
                ac = policy.step(obs)
                obs, reward, done, _ = env.step(ac)
                policy.get_reward(reward)
                ep_ret += reward
                ep_len += 1
            avg_reward = (avg_reward * j + ep_ret) / (j + 1)
            pbar.set_description(f"Epoch: {j}, avg_reward: {avg_reward:.2f}")
            pbar.update(1)

def evaluate(env,policy,num_evaluate_episodes,is_render):
    for j in range(num_evaluate_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0
        while not(done):
            if is_render:
                env.render()
                # sleep(0.025)
            # Take deterministic actions at test time 
            ac = policy.step(obs)
            obs, reward, done, _ = env.step(ac)
            ep_ret += reward
            ep_len += 1
        policy.log_mean("TestEpRet", ep_ret)
        policy.log_mean("TestEpLen", ep_len)
    policy.dumpkvs()
