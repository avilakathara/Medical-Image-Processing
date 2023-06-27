import os
import numpy as np
import matplotlib.pyplot as plt
from src.slice_select.cost import cost
from src.slice_select.lbfgs import lbfgs
from src.slice_select.optimization import get_optimal_slice, get_gradients, random_plane_normal, gradient_descent
from src.slice_select.pso import particle_swarm_optimization
import datetime


def evaluate_lbfgs(N = 30):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            test_arrs.append(np.load(folder_name + '/' + path))

    max_iterations = 20
    max_files = 24
    initializations_per_file = 1


    save_folder = f'evaluation_data/lbfgs/lbfgs_it{max_iterations}_inm{N}'
    os.mkdir(save_folder)

    lbfgs_scores = []
    # lbfgs_scores = np.empty((0, max_iterations), float)
    for array_index, test_arr in enumerate(test_arrs):
        if array_index >= max_files:
            break
        for i in range(initializations_per_file):
            unmaximized_scores = []
            for j in range(N):
                print("array {}, initialization {}, part {} of maximum".format(array_index, i, j))

                # max_iterations-1 is necessary because the initial position is added as a '0th iteration'.
                costs, x = lbfgs(test_arr, max_iterations - 1)

                costs = np.pad(np.array(costs), (0, max_iterations - len(costs)), 'edge')
                unmaximized_scores.append(costs)

            max_scores = np.max(np.array(unmaximized_scores), axis=0)
            lbfgs_scores.append(max_scores)

            save_path = f'{save_folder}/{array_index}_{i}'
            #np.save(save_path, max_scores)

            print(datetime.datetime.now())

    lbfgs_scores = np.array(lbfgs_scores)
    lbfgs_average = np.mean(lbfgs_scores, axis=0)

    save_path = f'evaluation_data/lbfgs/lbfgs_it{max_iterations}_f{max_files}_ina{initializations_per_file}_inm{N}'
    np.save(save_path, lbfgs_average)

    plt.plot(lbfgs_average)

    plt.title(
        f'L-BFGS-B on {max_files} files using {initializations_per_file} initializations for randomness and {N} initializations for the maximum',
        loc='center',
        wrap=True)

    plt.show()
    print(datetime.datetime.now())


def evaluate_pso(N=30,iterations=50):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            # print(folder_name + '/' + path)
            test_arrs.append(np.load(folder_name + '/' + path))

    max_files = 24
    initializations_per_file = 1

    particles = N

    omega = 0.8
    c1 = 0.2
    c2 = 0.2


    save_folder = f'evaluation_data/pso/pso_o{omega}_c1={c1}_c2={c2}_it{iterations}_p{particles}'
    os.mkdir(save_folder)

    pso_scores = []
    for array_index, test_arr in enumerate(test_arrs):
        if array_index >= max_files:
            break
        for i in range(initializations_per_file):
            print(f"file {array_index}, initialization {i}")
            position, score, best_solutions = particle_swarm_optimization(test_arr, iterations, particles, omega, c1,
                                                                          c2)
            pso_scores.append(best_solutions)
            save_path = f'{save_folder}/{array_index}_{i}'
            #np.save(save_path, best_solutions)
            print(datetime.datetime.now())

    pso_scores = np.array(pso_scores)
    pso_average = np.mean(pso_scores, axis=0)

    np.save(save_folder, pso_average)

    title = f'Particle swarm optimization with {initializations_per_file} initializations per file with {max_files} ' + \
            f'files.\nRunning for {iterations} iterations with {particles} particles.\n Omega={omega}, c1={c1}, c2={c2}'
    plt.title(title)
    # plt.figure(dpi=1200)
    plt.plot(pso_average)
    plt.show()


def evaluate_gd(N = 5, iterations = 250, point_step_size=0.05, normal_step_size=1e-5):
    print(datetime.datetime.now())
    folder_name = 'uncertainty_fields'
    test_paths = os.listdir(folder_name)
    test_arrs = []
    for path in test_paths:
        if path.endswith('.npy'):
            test_arrs.append(np.load(folder_name + '/' + path))

    gd_scores = []
    max_files = 24
    initializations_per_file = 1

    save_folder = f'evaluation_data/gd/gd_pss{point_step_size}_nss{normal_step_size}_it{iterations}_inm{N}'
    #os.mkdir(save_folder)

    for array_index, test_arr in enumerate(test_arrs):

        if array_index >= max_files:
            break
        gradients = get_gradients(test_arr)

        for i in range(initializations_per_file):
            unmaximized_scores = []
            print("----------------------------------------")
            for j in range(N):
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

            save_path = f'{save_folder}/{array_index}_{i}'
            #np.save(save_path, max_scores)
            print(datetime.datetime.now())

    gd_scores = np.array(gd_scores)
    gd_average = np.mean(gd_scores, axis=0)

    save_path = f'evaluation_data/gd/gd_pss{point_step_size}_nss{normal_step_size}_it{iterations}_f{max_files}_ina{initializations_per_file}_inm{N}'
    np.save(save_path, gd_average)

    plt.plot(gd_average)
    plt.title(f'gradient descent with pss {point_step_size} and nss {normal_step_size} using {initializations_per_file}\
 initializations for randomness and {N} initializations for the maximum', loc='center', wrap=True)

    plt.show()


def find_step_sizes():
    point_step_sizes = [0.05]
    normal_step_sizes = [1e-5, 0.0]
    for point_step_size in point_step_sizes:
        for normal_step_size in normal_step_sizes:
            print('point step size {}, normal step size {}'.format(point_step_size, normal_step_size))
            evaluate_gd(point_step_size, normal_step_size)




if __name__ == "__main__":
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('Particle swarm optimization')
    print('N = 30')
    print('')
    evaluate_lbfgs(N=30)
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('Particle swarm optimization')
    print('N = 12')
    print('')
    evaluate_pso(N=12,iterations=50)



    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('Gradient descent')
    print('N = 12')
    print('')
    evaluate_gd(N=12,iterations=250)
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('Gradient descent')
    print('N = 5')
    print('')
    evaluate_gd(N=5,iterations=250)

    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('L-BFGS')
    print('N = 30')
    print('')
    evaluate_lbfgs(N=30)
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('//////////////////////////////////////////////////')
    print('L-BFGS')
    print('N = 12')
    print('')
    evaluate_lbfgs(N=12)


    print("Done")
