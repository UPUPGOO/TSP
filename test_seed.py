# -*- coding: utf-8 -*-
"""
experiment on different seed. For more detail, please reference to Readme.md
"""
import numpy as np
from sa import SA

data = np.loadtxt('location.txt', delimiter=',', usecols=[1, 2])
seed = None
sa = SA(data, T=200, anneal_rate=0.95, final_T=0.001, inner_iters=120, random_init=True)
seeds = np.arange(500, 1000)
with open('./experiment/2.61_seed.txt', 'w') as f:
    for seed in seeds:
        np.random.seed(seed)
        sa.reset()
        log = sa.train(mode='reverse')
        s = 'seed={}\ttime={:.3f}\tdistance={}'.format(seed, log['time'], log['dis'])
        print(s)
        print(s, file=f, flush=True)
        f.flush()
