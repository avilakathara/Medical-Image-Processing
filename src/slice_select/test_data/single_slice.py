import numpy as np

n = 20

arr = np.zeros((n, n, n))
middle = n // 2

arr[middle - 1: middle + 1, :, :] = 0.5
arr[middle, :, :] = 1

print(arr)
print(arr[10])
np.save('slice_' + str(n) + '.npy', arr)
