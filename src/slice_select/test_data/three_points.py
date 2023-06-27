import numpy as np
import scipy
import matplotlib.pyplot as plt


def generate_image():
    points = np.zeros((50, 50, 50), dtype=np.float32)

    p1 = (8, 10, 37)
    p2 = (30, 7, 12)
    p3 = (40, 30, 25)

    points[p1] = 255
    points[p2] = 255
    points[p3] = 255

    points = scipy.ndimage.gaussian_filter(points, sigma=10)
    points = points / np.max(points)

    plt.imshow(points[35], cmap='gray', vmin=0, vmax=1)
    plt.show()

    np.save('three_points', points)


if __name__ == "__main__":
    generate_image()
