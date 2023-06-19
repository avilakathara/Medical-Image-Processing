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

    max_files = 8
    initializations_per_file = 4
    initializations_for_max = 30

    lbfgs_scores = []
    # lbfgs_scores = np.empty((0, max_iterations), float)
    for array_index, test_arr in enumerate(test_arrs):
        if array_index >= max_files:
            break
        for i in range(initializations_per_file):
            unmaximized_scores = []
            for j in range(initializations_for_max):
                print("array {}, initialization {}, part {} of maximum".format(array_index, i, j))

                # max_iterations-1 is necessary because the initial position is added as a '0th iteration'.
                costs = lbfgs(test_arr, max_iterations - 1)

                costs = np.pad(np.array(costs), (0, max_iterations - len(costs)), 'edge')
                unmaximized_scores.append(costs)

            max_scores = np.max(np.array(unmaximized_scores), axis=0)
            lbfgs_scores.append(max_scores)
            print(datetime.datetime.now())

    lbfgs_scores = np.array(lbfgs_scores)
    lbfgs_average = np.mean(lbfgs_scores, axis=0)

    save_path = f'evaluation_data/lbfgs/lbfgs_it{max_iterations}_f{max_files}_ina{initializations_per_file}_inm{initializations_for_max}'
    np.save(save_path, lbfgs_average)

    plt.plot(lbfgs_average)

    plt.title(
        f'L-BFGS-B on {max_files} files using {initializations_per_file} initializations for randomness and {initializations_for_max} initializations for the maximum',
        loc='center',
        wrap=True)

    plt.show()
    print(datetime.datetime.now())


def evaluate_pso():
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            # print(folder_name + '/' + path)
            test_arrs.append(np.load(folder_name + '/' + path))

    max_files = 8
    initializations_per_file = 4

    iterations = 250
    particles = 30

    omega = 0.8
    c1 = 0.2
    c2 = 0.2

    pso_scores = []
    for array_index, test_arr in enumerate(test_arrs):
        if array_index >= max_files:
            break
        for i in range(initializations_per_file):
            print(f"file {array_index}, initialization {i}")
            position, score, best_solutions = particle_swarm_optimization(test_arr, iterations, particles, omega, c1,
                                                                          c2)
            pso_scores.append(best_solutions)
        print(datetime.datetime.now())

    pso_scores = np.array(pso_scores)
    pso_average = np.mean(pso_scores, axis=0)

    save_path = f'evaluation_data/pso/pso_o{omega}_c1={c1}_c2={c2}_it{iterations}_f{max_files}_ina{initializations_per_file}_p{particles}'
    np.save(save_path, pso_average)

    title = f'Particle swarm optimization with {initializations_per_file} initializations per file with {max_files} ' + \
            f'files.\nRunning for {iterations} iterations with {particles} particles.\n Omega={omega}, c1={c1}, c2={c2}'
    plt.title(title)
    # plt.figure(dpi=1200)
    plt.plot(pso_average)
    plt.show()


def evaluate_gd(point_step_size=0.5, normal_step_size=0.02):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            # print(folder_name + '/' + path)
            test_arrs.append(np.load(folder_name + '/' + path))

    gd_scores = []
    max_files = 8
    initializations_per_file = 4
    initializations_for_max = 5
    iterations = 250

    for array_index, test_arr in enumerate(test_arrs):

        if array_index >= max_files:
            break
        gradients = get_gradients(test_arr)

        for i in range(initializations_per_file):
            unmaximized_scores = []
            for j in range(initializations_for_max):
                print("----------------------------------------")
                print("array {}, initialization {}, part {} of maximum".format(array_index, i, j))
                start_pos = np.random.uniform(np.zeros(3), test_arr.shape)
                start_normal = random_plane_normal()
                # print("start normal:")
                # print(start_normal)

                current_pos, current_normal, costs = gradient_descent(test_arr, start_pos, start_normal,
                                                                      point_step_size,
                                                                      normal_step_size, gradients, it=iterations)
                # print("end normal:")
                # print(current_normal)
                unmaximized_scores.append(costs)

            max_scores = np.max(np.array(unmaximized_scores), axis=0)
            gd_scores.append(max_scores)
            print(datetime.datetime.now())

    gd_scores = np.array(gd_scores)
    gd_average = np.mean(gd_scores, axis=0)

    save_path = f'evaluation_data/gd/gd_pss{point_step_size}_nss{normal_step_size}_it{iterations}_f{max_files}_ina{initializations_per_file}_inm{initializations_for_max}'
    np.save(save_path, gd_average)

    plt.plot(gd_average)
    plt.title(f'gradient descent with pss {point_step_size} and nss {normal_step_size} using {initializations_per_file}\
 initializations for randomness and {initializations_for_max} initializations for the maximum', loc='center', wrap=True)

    plt.show()


def find_step_sizes():
    point_step_sizes = [0.05]
    normal_step_sizes = [1e-5, 0.0]
    for point_step_size in point_step_sizes:
        for normal_step_size in normal_step_sizes:
            print('point step size {}, normal step size {}'.format(point_step_size, normal_step_size))
            evaluate_gd(point_step_size, normal_step_size)




if __name__ == "__main__":
    # find_step_sizes()
    # evaluate_gd(0.05, 1e-5)
    # evaluate_pso()
    # evaluate_lbfgs(250)
    print("Done")
