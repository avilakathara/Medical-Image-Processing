import numpy as np
import cv2 as cv2

# takes the uncertainty field and returns the plane (slice) with the highest uncertainty


best_of_planes = []


# Input: 3D numpy array
# Output: a 2D array representing the indexes in the 3D array

# def get_optimal_slice():
#     return
#
#
# def multi_gradient_descent(uncertainty, iterations, step_size, multi):
#     highest_value = -1
#     for i in range(0, multi):
#         normal = np.random.rand(3) * 360
#         temp = gradient_descent(uncertainty, iterations, step_size, normal)
#         if highest_value < temp:
#             highest_value = temp
#     return highest_value
#
#
# # Input: a 3D numpy array with values between 0 and 1
# def gradient_descent(uncertainty, iterations, step_size, normal):
#     # instantiate random plane: random position within current segmentation
#     current_pos = 56
#     for i in range(iterations):
#         # calculate gradient (uncertainty of one step above - one step below)
#         gradient = calculate(current_pos+1) - calculate(current_pos-1)
#         current_pos += gradient * step_size  # move to higher uncertainty, repeat
#
#     uncertainty_value = calculate(current_pos)
#
#     return uncertainty_value, current_pos
#
#
# # Sum over a 2d array
# def calculate(pos):
#     return 0


def method(uncertainty):
    x, _, _ = uncertainty.shape
    highest_value = -1
    highest_x = -1
    step_size = 1
    v = -1
    for i in range(0, 1):
        # x = np.random.rand(1) * x
        x = int(np.random.uniform(1, x - 1))
        temp_x, temp, temp_v = gradient_descent2(uncertainty, x, step_size)
        if highest_value < temp:
            highest_value = temp
            highest_x = temp_x
            v = temp_v
    return highest_value, highest_x, v


def gradient_descent2(uncertainty, x, step_size):
    current_pos = x
    for i in range(250):
        #print(current_pos)
        gradient_difference = func(uncertainty[current_pos + 2, :, :]) - func(uncertainty[current_pos + 1, :, :])
        if gradient_difference > 0:
            current_pos += step_size  # difference in position should or shouldn't be dependant on gradient?
        else:
            current_pos -= step_size
    return current_pos + 1, func(uncertainty[current_pos + 1, :, :]), x


def func(plane):
    gradient_image = cv2.Laplacian(plane, cv2.CV_64F)
    arr = abs(gradient_image * plane)
    return np.sum(arr)


if __name__ == "__main__":
    # arr = np.arange(20)
    # arr = arr.reshape((-1,1,1))
    # arr_2d = np.repeat(arr, 20,axis=1)
    # arr_3d = np.repeat(arr_2d, 20,axis=2)
    # test_arr = 1 - np.abs(arr_3d/10 - 1)
    # noise = np.random.rand(20,20,20)/10
    # test_arr = test_arr + noise
    # np.save('my_array.npy', test_arr)
    test_arr = np.load('my_array.npy')

    print("Mean: " + str(np.mean(test_arr, (1, 2))))

    arr = []

    for x in range(1, 20):
        arr.append(func(test_arr[x]))

    print(arr)

    h_v, h_x, v = method(test_arr)
    print(h_v)
    print(h_x)
    print(v)
    # print(test_arr[h_x + 1])
    # print(test_arr[h_x - 1])
    # print(np.sum(test_arr[1]))
    # print(np.sum(test_arr[4]))
