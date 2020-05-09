# -*- coding: utf-8 -*- 
import numpy as np
import math
import random
from time import time
import matplotlib.pyplot as plt
from bisect import bisect


class SA:
    def __init__(self,
                 data,
                 T=120,
                 anneal_rate=0.99,
                 final_T=0.001,
                 inner_iters=500,
                 random_init=False):
        self.T = T
        assert 0 < anneal_rate < 1, 'anneal_rate must be between 0 and 1'
        self.gamma = anneal_rate
        self._T = final_T
        self.inner_iters = inner_iters
        self.n = len(data)
        self.data = data
        self.dist = self.get_distance_dist(data)
        self.random_init = random_init
        self.reset()

    def reset(self):
        self.temp = self.T
        self._cnt = 0
        self._invalid_cnt = 0
        self.init_path(self.random_init)
        self.best_path, self.best_dis = self._path, self._dis
        self.dis_lst = [self._dis]

    def init_path(self, random_path=False):
        self._path = np.arange(self.n)
        if random_path:
            np.random.shuffle(self._path[1:])
        self._dis = self.cal_path(self._path)
        return self._path

    def step(self, mode='reverse', greedy=False):
        self._cnt += 1
        if mode == 'cross':
            new_path = self.cross(self._path.copy())
        elif mode == 'reverse':
            new_path = self.reverse(self._path.copy())
        else:
            raise ValueError

        new_dis = self.cal_path(new_path)
        delta_dis = new_dis - self._dis
        if new_dis < self.best_dis:
            self.best_path = new_path
            self.best_dis = new_dis
        if delta_dis < 0 or (not greedy and np.random.random() < self.p(delta_dis)):
            self._path = new_path
            self._dis = new_dis
            self.dis_lst.append(self._dis)
        else:
            return False
        return True

    def train(self, iters=None, mode='reverse', anneal_mode=0, greedy=False, verb=False):
        iters = self.inner_iters if iters is None else int(iters)
        time_lst = []
        while True:
            self._invalid_cnt = 0
            t0 = time()
            for i in range(iters):
                if not self.step(mode, greedy):
                    self._invalid_cnt += 1
                    if self._invalid_cnt > min(int(iters * 0.5), 100):
                        break
            t1 = time()
            if verb:
                print('Epi{}, Time {:.2f}, current best distance {}'.format(len(time_lst), t1 - t0, self.best_dis))
            time_lst.append(t1 - t0)
            if anneal_mode == 0:
                self.temp *= self.gamma
            elif anneal_mode == 1:
                self.temp /= math.log(1 + self._cnt)
            elif anneal_mode == 2:
                self.temp /= (1 + self._cnt)
            else:
                raise ValueError
            if self.temp <= self._T:
                break
        if verb:
            print(f'Total time:{np.sum(time_lst)},best distance {self.best_dis}')
        return {'path': self.best_path,
                'dis': self.best_dis,
                'time': np.sum(time_lst)
                }

    def reverse(self, path):
        i, j = np.random.choice(range(1, self.n), 2)
        if i < j:
            path[i:j + 1] = path[j:i - 1:-1]
        elif i > j:
            path[j:i + 1] = path[i:j - 1:-1]
        return path

    def cross(self, path):
        i, j = np.random.choice(range(1, self.n), 2)
        path[i], path[j] = path[j], path[i]
        return path

    def _compare_path(self, p1, p2):
        return np.sum(p1 == p2) / len(p1)

    def compare_path(self, p1, p2=None):
        if len(p1.shape) > 1 and p2 is None:
            return np.array([self._compare_path(m, n) for m in p1 for n in p1]).reshape(-1, p1.shape[0])
        elif len(p1.shape) == 1 and p2 is None:
            raise ValueError
        elif len(p1.shape) == len(p2.shape) and len(p1.shape) > 1:
            return np.array([self._compare_path(m, n) for m, n in zip(p1, p2)])
        elif len(p1.shape) == len(p2.shape) and len(p1.shape) == 1:
            return self._compare_path(p1, p2)
        elif len(p1.shape) > 1 and len(p2.shape) == 1:
            return np.array([self._compare_path(m, p2) for m in p1])
        elif len(p2.shape) > 1 and len(p1.shape) == 1:
            return np.array([self._compare_path(m, p1) for m in p2])
        else:
            raise ValueError

    def cal_path(self, path):
        dist = self.dist[path[-1], path[0]]
        for i in range(len(path) - 1):
            dist += self.dist[path[i], path[i + 1]]
        return dist

    def get_distance_dist(self, data):
        return np.array([np.linalg.norm(m - n) for m in data for n in data]).round().astype(np.int).reshape(-1, len(data))

    def p(self, delta):
        p = math.exp(-delta / self.temp)
        return p

    def plot_dis(self, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.dis_lst)
        ax.set_xlabel('num')
        ax.set_ylabel('distance')
        title = 'best distance:' + str(np.min(self.dis_lst))
        if 'title' in kwargs:
            title += '\t' + kwargs['title']
        ax.set_title(title)
        plt.show()

    def plot_path(self, bestPath):
        ax = plt.subplot(111, aspect='equal')
        ax.plot(self.data[:, 0], self.data[:, 1], 'x', color='blue')
        for i, loc in enumerate(self.data):
            ax.text(loc[0], loc[1], str(i))
        ax.plot(self.data[bestPath, 0], self.data[bestPath, 1], color='red')
        plt.show()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p
        self._dis = self.cal_path(p)

    @property
    def dis(self):
        return self._dis


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([m - n for m in a for n in a])
    print(b)
