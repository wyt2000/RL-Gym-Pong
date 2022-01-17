import os
from importlib import import_module
import gym

import sys
sys.path.insert(0, os.getcwd())

from src.utils.misc_utils import get_params_from_file
from src.utils.pipeline import evaluate, train
from src.alg.RL_alg import RL_alg
from src.alg.PB18111684.PB18111684 import PB18111684

def init_env_policy(alg, env_name):
    env = gym.make(env_name)
    print('Env:',env_name)
    
    policy = alg(env.observation_space,env.action_space)
    print('Load alg')

    return env, policy

if __name__ == '__main__':
    main_params = get_params_from_file('configs.main_setting',params_name='params')
    if os.path.exists('./score.csv'):
        os.remove('./score.csv')
    for alg in RL_alg.__subclasses__():
        env, policy = init_env_policy(alg, main_params['env_name'])
        train(env,policy,**main_params['train'])
        del env, policy





