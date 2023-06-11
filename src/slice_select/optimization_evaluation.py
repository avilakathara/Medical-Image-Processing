import os
import numpy as np
import matplotlib.pyplot as plt
from src.slice_select.cost import cost
from src.slice_select.lbfgs import lbfgs
from src.slice_select.optimization import get_optimal_slice, get_gradients, random_plane_normal, gradient_descent
from src.slice_select.pso import particle_swarm_optimization
import datetime


def evaluate_lbfgs(max_iterations):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            print(folder_name + '/' + path)
            test_arrs.append(np.load(folder_name + '/' + path))

    lbfgs_scores = np.empty((0,max_iterations), float)
    for test_arr in test_arrs:
        for i in range(30):
            print("initialization {}".format(i))

            costs = lbfgs(test_arr, max_iterations)

            costs = np.pad(np.array(costs), (0, max_iterations - len(costs)), 'edge')
            lbfgs_scores = np.append(lbfgs_scores, [costs], axis=0)
            print(lbfgs_scores)



    lbfgs_average = np.mean(lbfgs_scores, axis=0)

    plt.plot(lbfgs_average)
    print(lbfgs_scores.shape)
    print(lbfgs_average)
    plt.title('l-bfgs-b')

    plt.show()
    print(datetime.datetime.now())


def evaluate_gd(point_step_size=0.5, normal_step_size=0.02):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            print(folder_name + '/' + path)
            test_arrs.append(np.load(folder_name + '/' + path))

    test_arr = np.load('uncertainty_fields/0522c0159_o3_i1.npy')
    # test_arr = np.load('test_data/sphere_20.npy')

    # pso_scores = []
    # for i in range(10):
    #     print("initialization {}".format(i))
    #     position, score, best_solutions = particle_swarm_optimization(test_arr, 20, 15, 0.8, 0.2, 0.2)
    #     pso_scores.append(best_solutions)
    #
    # pso_scores = np.array(pso_scores)
    # pso_average = np.mean(pso_scores, axis=0)
    # plt.plot(pso_average)
    gd_scores = []
    for test_arr in test_arrs:
        gradients = get_gradients(test_arr)

        for i in range(3):
            print("initialization {}".format(i))
            start_pos = np.random.uniform(np.zeros(3), test_arr.shape)
            start_normal = random_plane_normal()

            current_pos, current_normal, costs = gradient_descent(test_arr, start_pos, start_normal, point_step_size,
                                                                  normal_step_size, gradients)
            gd_scores.append(costs)

    gd_scores = np.array(gd_scores)
    gd_average = np.mean(gd_scores, axis=0)
    np.save('graph_data/gd_pss{}_nss{}'.format(point_step_size, normal_step_size), gd_average)
    plt.plot(gd_average)
    plt.title('gradient descent with pss {} and nss {}'.format(point_step_size, normal_step_size))

    plt.show()
    print(datetime.datetime.now())


def find_step_sizes():
    point_step_sizes = [1, 0.5, 0.25, 0.0625, 0, -0.0625, -0.25, -0.5, -1]
    normal_step_sizes = [0.5, 0.25, 0.1, 0.04, 0.001, 0, -0.001, -0.01, -0.04, -0.25, -0.5]
    for point_step_size in point_step_sizes:
        for normal_step_size in normal_step_sizes:
            print('point step size {}, normal step size {}'.format(point_step_size, normal_step_size))
            evaluate_gd(point_step_size, normal_step_size)


if __name__ == "__main__":
    # find_step_sizes()
    evaluate_lbfgs(20)
