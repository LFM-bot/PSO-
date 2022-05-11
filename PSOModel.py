import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PSO(object):
    def __init__(self, func_udefine, sizepop=50, iter=300, rangepop=(-10, 10)):
        self.func_udefine = func_udefine
        self.iter = iter  # 迭代次数
        self.w = 0.8  # 惯性权重
        self.lr = (0.6, 0.3)  # 粒子群个体和社会的学习因子，即加速常数
        self.sizepop = sizepop  # 种群规模
        self.rangepop = rangepop  # 粒子的位置的范围限制,x、y方向的限制相同
        self.rangespeed = (-0.5, 0.5)  # 速度限制
        self.px = []  # 当前全局最优解
        self.py = []

    def func(self, x):
        # x输入粒子位置
        # y 粒子适应度值
        y = self.func_udefine(x[0], x[1])

        return y

    def initpopvfit(self, sizepop):
        pop = np.zeros((sizepop, 2))
        v = np.zeros((sizepop, 2))
        fitness = np.zeros(sizepop)

        for i in range(sizepop):
            pop[i] = [np.random.rand() * (self.rangepop[1] - self.rangepop[0]),
                      np.random.rand() * (self.rangepop[1] - self.rangepop[0])]
            v[i] = [np.random.rand() * (self.rangepop[1] - self.rangepop[0]),
                    np.random.rand() * (self.rangepop[1] - self.rangepop[0])]
            fitness[i] = self.func(pop[i])

        return pop, v, fitness

    def getinitbest(self, fitness, pop):
        # 群体最优的粒子位置及其适应度值
        gbestpop, gbestfitness = pop[fitness.argmax()].copy(), fitness.max()
        # 个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似

        pbestpop, pbestfitness = pop.copy(), fitness.copy()

        return gbestpop, gbestfitness, pbestpop, pbestfitness

    def run(self):
        """
        通过循环迭代，不断的更新粒子的位置和速度，根据新粒子的适应度值更新个体和群体的极值
        Returns
        -------
        result: np.Array, [iter_num]
            best global solution for each iteration.
        px: np.Array, [iter_num]
            best x for each iteration
        py: np.Array, [iter_num]
            best y for each iteration

        """
        pop, v, fitness = self.initpopvfit(self.sizepop)
        gbestpop, gbestfitness, pbestpop, pbestfitness = self.getinitbest(fitness, pop)

        result = np.zeros(self.iter)
        for i in range(self.iter):
            # 速度更新
            for j in range(self.sizepop):
                v[j] = self.w * v[j] + self.lr[0] * np.random.rand() * (pbestpop[j] - pop[j]) + self.lr[
                    1] * np.random.rand() * (
                               gbestpop - pop[j])
            v[v < self.rangespeed[0]] = self.rangespeed[0]
            v[v > self.rangespeed[1]] = self.rangespeed[1]

            # 粒子位置更新
            for j in range(self.sizepop):
                pop[j] += v[j]
            pop[pop < self.rangepop[0]] = self.rangepop[0]
            pop[pop > self.rangepop[1]] = self.rangepop[1]

            # 适应度更新
            for j in range(self.sizepop):
                fitness[j] = self.func(pop[j])

            for j in range(self.sizepop):
                if fitness[j] > pbestfitness[j]:
                    pbestfitness[j] = fitness[j]
                    pbestpop[j] = pop[j].copy()

            if pbestfitness.max() > gbestfitness:
                gbestfitness = pbestfitness.max()
                gbestpop = pop[pbestfitness.argmax()].copy()

            result[i] = gbestfitness
            self.px.append(gbestpop[0])
            self.py.append(gbestpop[1])

        return result, self.px, self.py

    def drawPaht(self, X, Y, Z, px, py, pz):
        """
        绘图
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.title("PSO")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', cmap='hot')
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.plot(px, py, pz, 'r.')  # 绘点x
        plt.show()
