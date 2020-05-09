# -*- coding: utf-8 -*-
"""
experiment on different initial temperature. For more detail, please reference to Readme.md
"""
import numpy as np
from sa import SA

data = np.loadtxt('location.txt', delimiter=',', usecols=[1, 2])
seed = None
sa = SA(data, T=200, anneal_rate=0.95, final_T=0.001, inner_iters=120, random_init=True)
Ts = [10, 30, 50, 100, 150, 200, 240, 300]
n = 5
with open('./experiment/2.1_initT.txt', 'w') as f:
    for T in Ts:
        t = 0
        dis = 0
        for i in range(n):
            sa.reset()
            sa.temp = T
            log = sa.train(mode='reverse')
            t += log['time']
            dis += log['dis']
        s='T={}\ttime={:.3f}\tdistance={}'.format(T, t / n, dis / n)
        print(s,file=f,flush=True)
        print(s)
