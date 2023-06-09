import numpy as np
import matplotlib.pyplot as plt
from src.slice_select.cost import cost
from src.slice_select.optimization import get_optimal_slice, get_gradients, random_plane_normal, gradient_descent
from src.slice_select.pso import particle_swarm_optimization

if __name__ == "__main__":
    test_arr = np.load('test_data/uncertainty.npy')
    test_arr = test_arr[:, 250:350, 250:350]

    pso_scores = []
    # for i in range(10):
    #     print("initialization {}".format(i))
    #     position, score, best_solutions = particle_swarm_optimization(test_arr, 20, 15, 0.8, 0.2, 0.2)
    #     pso_scores.append(best_solutions)
    #
    # pso_scores = np.array(pso_scores)
    # pso_average = np.mean(pso_scores, axis=0)
    # plt.plot(pso_average)

    gd_scores = []
    gradients = get_gradients(test_arr)
    for i in range(10):
        print("initialization {}".format(i))
        start_pos = np.random.uniform(np.zeros(3), test_arr.shape)
        start_normal = random_plane_normal()
        point_step_size = 0.5
        normal_step_size = 0.02

        current_pos, current_normal, costs = gradient_descent(test_arr, start_pos, start_normal, point_step_size, normal_step_size, gradients)
        gd_scores.append(costs)

    gd_scores = np.array(gd_scores)
    gd_average = np.mean(gd_scores, axis=0)
    plt.plot(gd_average)


    plt.show()