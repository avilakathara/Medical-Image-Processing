
import math
import numpy as np
import numpy.linalg as nl
import scipy.ndimage as sn

binary_labels = []

def calculate_uncertainty_fields(image, label, prob, combine_fields=True):
    """
    Method for calculating uncertainty fields

    @param image:   3D image of intensity values
    @param label:   3D image of segmentation (0 for background, 1 for organ 1, etc.)
    @param prob:    1D + 3D arrays containing classification probabilities of each voxels, for each labels

    @param combine_fields: if True, combine fields into 1 3D uncertainty field

    @return:        (1D +) 3D arrays containing uncertainty values of each voxels, for each labels
    """

    # get image size; ps is the number of label types
    ps, px, py, pz = prob.shape

    # assure other 3D-arrays have same size
    lx, ly, lz = label.shape
    if lx != px or ly != py or lz != pz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + px + ", " + py + ", " + pz + "]"
                        + "but got "
                        + "[" + lx + " ," + ly + ", " + lz + "].")
    ix, iy, iz = image.shape
    if ix != px or iy != py or iz != pz:
        raise Exception("Inconsistent array sizes. Expected "
                        + "[" + px + ", " + py + ", " + pz + "]"
                        + "but got "
                        + "[" + ix + " ," + iy + ", " + iz + "].")

    # parameters for foreground & background distributions
    # [(mean_0, std_0), (mean_1, std_1), ...]
    print("Calculating distributions...")
    global binary_labels
    binary_labels = convert_to_binary(label, ps)
    # distributions = gaussian_foreground_background(image, label, ps)

    # distance maps (or transform)
    # [(distance_from_0, distance_from_1), (df0, df2), ...]
    print("Calculating distance maps...")
    # distance_maps = get_distance_maps(label, ps)

    # image gradient
    print("Calculating gradient...")
    # dx, dy, dz = np.gradient(image)

    # initialize output uncertainty field
    # output_field = np.zeros((ps - 1, px, py, pz))
    # if combine_fields:
    #     output_field = np.zeros((px, py, pz))
    output_field = []

    # vectorize functions
    vec_entropy_e = np.vectorize(entropy_energy)
    vec_regional_e = np.vectorize(regional_energy)

    # assert len(binary_labels) > 0
    # assert len(distributions) > 0

    # calculate uncertainty for each voxel
    for ll in range(1, ps):
        print("processing label: " + str(ll))
        u_e = vec_entropy_e(prob[ll])
        # u_r = vec_regional_e(image, binary_labels[ll],
        #                      distributions[0][0], distributions[0][1], distributions[1][0], distributions[1][1])
        output_field.append(u_e)

    # for x in range(px):
    #     print("Working... x = " + str(x))
    #     for y in range(py):
    #         for z in range(pz):
    #             for p in range(0, ps - 1):
    #                 u_e = entropy_energy(float(prob[p, x, y, z]))
    #                 u_b = boundary_energy((x, y, z), label[x, y, z], distance_maps[p], (dx, dy, dz))
    #
    #                 # SLOW (with regional energy)
    #                 # u_r = regional_energy(image[x, y, z], label[x, y, z], distributions[0], distributions[p])
    #                 # output_field[p, x, y, z] = 0.8 * u_e + 0.2 * u_r  # + 0.05 * u_b

    output_field = np.array(output_field)

    if combine_fields:
        return (output_field-np.min(output_field))/(np.max(output_field)-np.min(output_field))

    return output_field

# --- ENTROPY ENERGY

def entropy_energy(prob):
    t1 = 0.0
    if prob > 0.0:
        t1 = float(prob * math.log(prob, 2))
    t2 = 0.0
    if prob < 1.0:
        t2 = float((1.0 - prob) * math.log(1.0 - prob, 2))
    return -t1 - t2

# --- REGIONAL ENERGY ---

def regional_energy(intensity, bin_lab, bgm, bgs, fgm, fgs):
    fg = gaussian_approx(intensity, fgm, fgs)
    bg = gaussian_approx(intensity, bgm, bgs)

    if bin_lab:
        return gaussian_approx(intensity, fgm, fgs) / (fg + bg)
    return gaussian_approx(intensity, bgm, bgs) / (fg + bg)

def convert_to_binary(label, n):
    bin_labels = [None]
    for i in range(1, n):
        bin_labels.append((label == i))
    return bin_labels

def gaussian_foreground_background(image, label, n):
    ix, iy, iz = image.shape
    intensity_list = []
    for i in range(n):
        intensity_list.append([])

    for x in range(ix):
        for y in range(iy):
            for z in range(iz):
                intensity_list[label[x, y, z]].append(image[x, y, z])

    output_list = []
    for i in range(n):
        output_list.append((np.mean(intensity_list[i]), np.std(intensity_list[i])))
    return output_list

def gaussian_density(x, mean, std):
    if x == 0:
        return 0

    expo = -math.pow((x - mean) / std, 2) / 2
    coef = 1 / (std * math.sqrt(2 * math.pi))
    return coef * math.pow(math.e, expo)

def gaussian_approx(x, mean, std):
    return 0.6 * (1 + math.cos((x - mean) / std)) / (std * math.pi)

# --- BOUNDARY ENERGY ---

def boundary_energy(x_coord, label, dm, gradient, alpha=2):
    delta = soft_delta_func(distance_to_boundary(x_coord, label, dm))
    gradient_abs_pow = math.pow(voxel_norm(x_coord, gradient), alpha)
    return delta / (1 + gradient_abs_pow)

def distance_to_boundary(x_coord, label, dm):
    x, y, z = x_coord
    df0, df1 = dm
    if label == 0:
        # voxel is classified as 0, get distance from nearest 1
        return df1[x, y, z]
    else:
        # voxel is classified as 1, get distance from nearest 0
        return df0[x, y, z]

def voxel_norm(x_coord, gradient):
    x, y, z = x_coord
    dx, dy, dz = gradient
    return nl.norm(np.array([dx[x, y, z], dy[x, y, z], dz[x, y, z]]))

def soft_delta_func(x):
    return math.pow(math.e, -(x * x) / 2)

def get_distance_maps(label, n):
    lx, ly, lz = label.shape
    labels = []
    for i in range(1, n):
        labels.append(np.zeros((lx, ly, lz)))

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                for i in range(0, n-1):
                    if label[x, y, z] == i:
                        labels[i][x, y, z] = 1
                    else:
                        labels[i][x, y, z] = 0

    output_map = []
    for i in range(0, n-1):
        output_map.append(0)
    for i in range(0, n-1):
        output_map.append((
            sn.distance_transform_edt(labels[i]),
            sn.distance_transform_edt(np.invert(labels[i]))
        ))
    return output_map
