import numpy as np

# takes the uncertainty field and returns the plane (slice) with the highest uncertainty


best_of_planes = []


# Input: 3D numpy array
# Output: a 2D array representing the indexes in the 3D array

def get_optimal_slice():
    return


def multi_gradient_descent(uncertainty, iterations, step_size, multi):
    highest_value = -1
    for i in range(0, multi):
        normal = np.random.rand(3) * 360
        temp = gradient_descent(uncertainty, iterations, step_size, normal)
        if highest_value < temp:
            highest_value = temp
    return highest_value


# Input: a 3D numpy array with values between 0 and 1
def gradient_descent(uncertainty, iterations, step_size, normal):
    # instantiate random plane: random rotation, position within current segmentation
    current_pos = 56
    for i in range(iterations):
        # calculate gradient (uncertainty of one step above - one step below)
        gradient = calculate(current_pos+1) - calculate(current_pos-1)
        current_pos += gradient * step_size  # move to higher uncertainty, repeat

    uncertainty_value = calculate(current_pos)

    return uncertainty_value, current_pos


# Sum over a 2d array
def calculate(pos):
    return 0
