import numpy as np
import matplotlib.pyplot as plt
from PSOModel import PSO
import time
from function import f1, f2
import argparse


def main(config):
    seed = config.seed
    np.random.seed(seed)

    pop_size = config.pop
    multi_pop_list = config.mul_pop
    iter_num = config.iter
    func = config.func
    x_range = config.x_range  # same as y range
    y_range = config.y_range

    pso = PSO(func, sizepop=pop_size, iter=iter_num, rangepop=x_range)

    if multi_pop_list is None:
        result, px, py = pso.run()  # result: fitness for each particle
        best_iter = result.argmax()
        x_best = px[best_iter]
        y_best = py[best_iter]
        print('N=%d best value:%.4f  ' % (pop_size, result[best_iter]))
        print('best solution x:%.4f y:%.4f' % (x_best, y_best))
        plt.plot(result, label='N=50')
    else:
        results = []
        for pop_size in multi_pop_list:
            pso = PSO(func, sizepop=pop_size, iter=iter_num, rangepop=x_range)
            result, px, py = pso.run()  # result: fitness for each particle
            results.append(result)
            best_iter = result.argmax()
            x_best = px[best_iter]
            y_best = py[best_iter]
            print('N=%d best value :%.4f  ' % (pop_size, result[best_iter]))
            print('best solution x:%.4f y:%.4f' % (x_best, y_best))

        for i, result in enumerate(results):
            plt.plot(result, label='N=%d' % multi_pop_list[i])

    plt.title('curve')
    plt.xlabel('iter')
    plt.ylabel('object')
    plt.legend()
    plt.show()

    if config.plot_3d:
        sample_step = 0.05
        x_list = np.arange(x_range[0], x_range[-1], sample_step)
        y_list = np.arange(y_range[0], y_range[-1], sample_step)
        X, Y = np.meshgrid(x_list, y_list)
        Z = func(X, Y)
        pso.drawPaht(X, Y, Z, [x_best], [y_best], func(np.array([x_best]), np.array([y_best])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper Parameters')
    # Model
    parser.add_argument('--w', type=float, default=0.8, help='inertia weight factor')
    parser.add_argument('--c1', type=float, default=0.6, help='acceleration factor 1')
    parser.add_argument('--c2', type=float, default=0.3, help='acceleration factor 2')
    parser.add_argument('--iter', type=int, default=300, help='pso max iteration time')
    parser.add_argument('--pop', type=int, default=50, help='pso initial population size')
    # Experiment
    parser.add_argument('--mul_pop', type=list, default=[30, 50, 70],
                        help='multiple population size, if use certain pop size, set None')
    parser.add_argument('--plot_3d', type=bool, default=True, help='Draw 3D view')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='random seed')
    parser.add_argument('--func', default=f2, help='experiment function')
    parser.add_argument('--x_range', type=tuple, default=(-10, 10), help='variable x range')
    parser.add_argument('--y_range', type=tuple, default=(-10, 10), help='variable y range')
    config = parser.parse_args()

    main(config)

