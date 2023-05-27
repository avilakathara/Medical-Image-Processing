import numpy as np
n = 70
size = (n,n,n)
arr = np.ones(size)
center = np.array(size) // 2

max_distance = np.linalg.norm(center)

for index in np.ndindex(size):
    distance = np.linalg.norm(np.array(index) - center)
    value = 1 - (distance / max_distance)
    arr[index] = value

print(arr)

np.save(str(n)+'sphere.npy',arr)