# -*- coding: utf-8 -*-
"""
experiment on different iter of inner loop. For more detail, please reference to Readme.md
"""
import numpy as np
from sa import SA

data = np.loadtxt('location.txt', delimiter=',', usecols=[1, 2])
seed = None
sa = SA(data, T=200, anneal_rate=0.95, final_T=0.001, inner_iters=120, random_init=True)
Ls = [10, 50, 100, 150, 200]
n = 5
with open('./experiment/2.4_L.txt', 'w') as f:
    for L in Ls:
        t = 0
        dis = 0
        for i in range(n):
            sa.reset()
            sa.inner_iters = L
            log = sa.train(mode='reverse')
            t += log['time']
            dis += log['dis']
        s = 'L={}\ttime={:.3f}\tdistance={}'.format(L, t / n, dis / n)
        print(s, file=f, flush=True)
        print(s)
