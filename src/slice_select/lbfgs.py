import scipy
import numpy as np
from src.slice_select.cost import cost

costs = []


def lbfgs(uncertainty, iterations, maxls=30):
    costs = []
    # instantiate randomly
    upper_bounds = np.array([0, 0, 0, 180, 180])
    upper_bounds[0:3] = uncertainty.shape
    lower_bounds = np.zeros(5)

    initial_position = np.random.uniform(lower_bounds, upper_bounds)

    def add_cost(x):
        costs.append(cost(uncertainty, x[0:3], x[3], x[4]))


    add_cost(initial_position)

    bounds = [(lower_bounds[0], upper_bounds[0]),
              (lower_bounds[1], upper_bounds[1]),
              (lower_bounds[2], upper_bounds[2]),
              (lower_bounds[3], upper_bounds[3]),
              (lower_bounds[4], upper_bounds[4])]

    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=calculate_cost, x0=initial_position, fprime=None,
                                          args=[uncertainty], approx_grad=True, bounds=bounds,
                                          iprint=-1, maxiter=iterations,maxls=maxls,factr=0,epsilon=1,
                                          callback=add_cost, pgtol=0)

    return costs, x


def calculate_cost(x, uncertainty):
    return -cost(uncertainty, x[0:3], x[3], x[4])




if __name__ == "__main__":
    test_arr = np.load('uncertainty_fields/0522c0161_o3_i1.npy')
    print(test_arr.shape)
    res, x = lbfgs(test_arr, 20)
    print('----------------------------------')
    print(res)

