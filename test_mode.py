# -*- coding: utf-8 -*-
"""
experiment on different state transition mode. For more detail, please reference to Readme.md
"""
import numpy as np
from sa import SA

data = np.loadtxt('location.txt', delimiter=',', usecols=[1, 2])
seed = None
sa = SA(data, T=200, anneal_rate=0.95, final_T=0.001, inner_iters=120, random_init=True)
modes = ['cross', 'reverse']
n = 10
with open('./experiment/2.5_mode.txt', 'w') as f:
    for mode in modes:
        t = 0
        dis = 0
        for i in range(n):
            sa.reset()
            log = sa.train(mode=mode)
            t += log['time']
            dis += log['dis']
        s='mode={}\ttime={:.3f}\tdistance={}'.format(mode, t / n, dis / n)
        print(s,file=f, flush=True)
        print(s)
