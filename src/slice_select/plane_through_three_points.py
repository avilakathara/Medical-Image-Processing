import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.slice_select.cost import get_coordinates_alt, rotate_vector
from src.slice_select.lbfgs import lbfgs
from src.slice_select.optimization import gradient_descent, get_optimal_slice, get_gradients, random_plane_normal
from src.slice_select.pso import particle_swarm_optimization

p1 = (17, 21, 74)
p2 = (59, 15, 23)
p3 = (81, 60, 50)


def generate_image():
    size = 100
    points = np.zeros((size, size, size), dtype=np.float32)

    points[p1] = 255
    points[p2] = 255
    points[p3] = 255

    points = scipy.ndimage.gaussian_filter(points, sigma=15)
    points = points / np.max(points)

    plt.imshow(points[35], cmap='gray', vmin=0, vmax=1)
    plt.show()

    # np.save('test_data/three_points.npy', points)
    return points


def load_image():
    return np.load("test_data/three_points.npy")


def do_stuff():
    np.set_printoptions(threshold=100000)
    image = load_image()
    point = [59, 15, 23]
    normal = [0.575694, -0.608892, 0.545735]
    coordinates, indices = get_coordinates_alt(image.shape, point, normal)
    min_indices = np.min(indices, axis=0)
    indices -= min_indices
    max_indices = np.max(indices, axis=0)

    asdf = np.zeros((max_indices[0] + 1, max_indices[1] + 1, 3))
    slice = np.zeros((max_indices[0] + 1, max_indices[1] + 1))
    for i in range(coordinates.shape[0]):
        asdf[indices[i][0], indices[i][1]] = coordinates[i]

    asdf = asdf.astype(np.int16)
    for a in range(asdf.shape[0]):
        for b in range(asdf.shape[1]):
            coord = asdf[a, b]
            slice[a, b] = image[coord[0], coord[1], coord[2]]

    plt.imshow(slice, cmap='gray', vmin=0, vmax=1)
    plt.show()

def plot_plane_correct():
    point = p1

    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
    v2 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])

    true_normal = np.cross(v1, v2)
    plot_plane(point, true_normal, "Optimal plane", "optimal")

def plot_plane_lbfgs():
    image = load_image()

    costs, x = lbfgs(image, 25)

    normal = rotate_vector(np.array([1, 0, 0]), x[3], x[4])
    point = x[0:3]

    print(normal)
    print(point)
    plot_plane(point, normal, "L-BFGS-B", "lbfgs")

def plot_plane_pso():
    image = load_image()

    omega = 0.5
    c1 = 0.2
    c2 = 0.2
    position, score, best_solutions = particle_swarm_optimization(image, 30, 30, omega, c1,
                                                                  c2)

    print(position[3])
    print(position[4])
    normal = rotate_vector(np.array([1, 0, 0]), position[3], position[4])
    point = position[0:3]

    print(normal)
    print(point)
    plot_plane(point, normal, "Particle swarm optimization", "pso")

def plot_plane_gd():
    image = load_image()

    gradients = get_gradients(image)
    start_pos = np.random.uniform(np.zeros(3), image.shape)
    start_normal = random_plane_normal()
    # print("start normal:")
    # print(start_normal)

    point, normal, costs = gradient_descent(image, start_pos, start_normal,
                                                          0.05,
                                                          1e-5, gradients, it=250)
    print(normal)
    print(point)
    plot_plane(point, normal, "Gradient descent", "gradient_descent")



def plot_plane(point, normal, title, filename):
    # Extract coefficients from the normal vector and point
    A, B, C = normal
    D = -np.dot(normal, point)

    # Generate grid points
    x_range = np.linspace(0, 100, 100)
    y_range = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate z values using the plane equation
    Z = (-A * X - B * Y - D) / C

    # Create 3D plot
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=30, azim=20, roll=0)

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.2)

    # p1 = (17, 21, 74)
    # p2 = (59, 15, 23)
    # p3 = (81, 60, 50)
    ax.scatter([17, 59, 81],[21, 15, 60],[74, 23, 50], color='green')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(0,100)

    plt.title(title)

    # Display the plot
    plt.savefig(fname=f"graph_data/plane_{filename}", dpi=400)
    plt.show()





if __name__ == "__main__":
    #generate_image()
    #plot_plane_correct()
    #plot_plane_gd()
    # plot_plane_pso()
    plot_plane_lbfgs()
