import numpy as np
import matplotlib.pyplot as plt
from src.slice_select.cost import cost

# returns the best found uncertainty and the position where it was found
# Parameters:
#   uncertainty: the uncertainty field, a 3D array
#   iterations:  the amount of iterations the algorithm runs
#   particles:   the amount of particles initialized in the swarm
#   omega:       the inertia weight. Higher values mean more global exploration, lower
#                values mean more local exploitation
#   c1 and c2:   acceleration coefficients, determine the influence of the local best
#                and global best, respectively
def particle_swarm_optimization(uncertainty, iterations, particles, omega, c1, c2):
    # set upper bounds for coordinates and rotation [x,y,z,rot_x,rot_y]
    upper_bounds = np.array([0, 0, 0, 180, 180])
    upper_bounds[0:3] = uncertainty.shape
    lower_bounds = np.zeros(5)

    # randomly instantiate positions
    positions = np.random.uniform(lower_bounds, upper_bounds, (particles, 5))
    # randomly instantiate velocities
    velocities = np.random.uniform(-upper_bounds, upper_bounds, (particles, 5))

    # instantiate global bests
    global_best_position = np.zeros(5)
    global_best_uncertainty = 0

    # instantiate local bests
    best_positions = positions
    best_uncertainties = np.zeros(particles)
    for p in range(particles):
        position = positions[p]
        best_uncertainties[p] = cost(uncertainty, position[0:3], position[3], position[4])

    # for debugging
    best_solutions = []
    for i in range(iterations):
        print("iteration {}".format(i))
        for p in range(particles):
            position = positions[p]
            # randomly generate r1 and r2
            r1 = np.random.uniform(0, 1, 5)
            r2 = np.random.uniform(0, 1, 5)
            # apply equation 3
            velocity = omega * velocities[p] + c1 * r1 * (best_positions[p] - positions[p]) + c2 * r2 * (
                        global_best_position - positions[p])
            # apply velocity limiting
            velocity = np.clip(velocity, -upper_bounds/4, upper_bounds/4)
            # apply equation 4
            velocities[p] = velocity
            positions[p] = position + velocity

            # apply position limiting
            positions[p] = np.clip(positions[p], lower_bounds, upper_bounds)
            # update local best
            uncertainty_value = cost(uncertainty, position[0:3], position[3], position[4])
            if uncertainty_value > best_uncertainties[p]:
                best_uncertainties[p] = uncertainty_value
                best_positions[p] = positions[p]
                # update global best
                if uncertainty_value > global_best_uncertainty:
                    global_best_uncertainty = uncertainty_value
                    global_best_position = positions[p]
        best_solutions.append(global_best_uncertainty)
    return global_best_position, global_best_uncertainty, best_solutions





if __name__ == "__main__":
    test_arr = np.load('test_data/uncertainty.npy')

    test_arr = test_arr[:,200:400,200:400]
    position, score, best_solutions = particle_swarm_optimization(test_arr, 20, 20, 0.8, 0.2, 0.2)
    plt.plot(best_solutions)
    plt.show()
