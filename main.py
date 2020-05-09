# -*- coding: utf-8 -*- 
import numpy as np
from sa import SA
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Use simulated annealing to solve TSP problem')
parser.add_argument('--file_path', type=str, default='location.txt', help='data path to be solved')
parser.add_argument('--seed', type=int, default=800, help='one of [800,689,308,556,112] is good choice.(default:800)')
parser.add_argument('--init_T', type=float, default=200, help='initial temperature(default:200)')
parser.add_argument('--anneal_rate', type=float, default=0.95, help='initial temperature anneal rate(default:0.95)')
parser.add_argument('--final_T', type=float, default=0.001, help='final temperature(default:0.001)')
parser.add_argument('--inner_iters', type=float, default=120, help='inner loop  iterations(default:120)')
parser.add_argument('--random_init', action='store_false', default=True,
                    help='whether to initial solution randomly(default:True)')
parser.add_argument('--mode', type=str, default='reverse',
                    help='the method of get new solution. It must be one of [reverse,cross](default:reverse)')
parser.add_argument('--anneal_mode', type=int, default=0, help='anneal mode. It must be one of [0,1,2](default:0)')
ars = parser.parse_args()

np.random.seed(ars.seed)  # set random seed for reproducing experiment results
data = np.loadtxt(ars.file_path, delimiter=',', usecols=[1, 2])  # load data
# init algorithm
sa = SA(data, T=ars.init_T, anneal_rate=ars.anneal_rate, final_T=ars.final_T, inner_iters=ars.inner_iters, random_init=ars.random_init)
# run algorithm
log = sa.train(mode=ars.mode, anneal_mode=ars.anneal_mode)
# log information
t = log['time']
dis = log['dis']
path = log['path']
dis_lst = deepcopy(sa.dis_lst)
print('time:{:.3f}\tbest distance:{}'.format(t, dis))
print('Now you can use test.exe to verify solution.')
# save best result. So you can excute test.exe to verify the solution
np.savetxt('city.txt', path + 1, fmt='%d', newline=',')
# plot distance over time
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(dis_lst)
ax.set_xlabel('num')
ax.set_ylabel('distance')
ax.set_title('best distance:' + str(dis))
# plot best path. It is similar to the result of running test.exe
ax = fig.add_subplot(212, aspect='equal')
ax.plot(data[:, 0], data[:, 1], 'x', color='blue')
for i, loc in enumerate(data):
    ax.text(loc[0], loc[1], str(i))
ax.plot(data[path, 0], data[path, 1], color='red')
plt.show()
