from collections import defaultdict
import os
from typing_extensions import ParamSpecArgs

class RL_alg(object):
    def __init__(self):
        self.team = []
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)

    def step(self):
        raise NotImplementedError

    def log_mean(self,key,value):
        pass

    def dumpkvs(self):
        f = open('./score.csv', 'a+t')
        f.close()


       