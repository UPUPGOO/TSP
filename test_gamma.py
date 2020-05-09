# -*- coding: utf-8 -*-
"""
experiment on different anneal rate(gamma). For more detail, please reference to Readme.md
"""
import numpy as np
from sa import SA

data = np.loadtxt('location.txt', delimiter=',', usecols=[1, 2])
seed = None
sa = SA(data, T=200, anneal_rate=0.95, final_T=0.001, inner_iters=120, random_init=True)
gammas = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
n = 5
with open('./experiment/2.3_gamma.txt', 'w') as f:
    for gamma in gammas:
        t = 0
        dis = 0
        for i in range(n):
            sa.reset()
            sa.gamma = gamma
            log = sa.train(mode='reverse')
            t += log['time']
            dis += log['dis']
        s = 'gamma={}\ttime={:.3f}\tdistance={}'.format(gamma, t / n, dis / n)
        print(s, file=f, flush=True)
        print(s)
