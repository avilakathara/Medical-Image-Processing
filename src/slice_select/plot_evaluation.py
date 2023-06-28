import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os


def plot_graph(a1, a2, a3, tick_freq, max_iter, min_u, max_u, title, filename):
    saved_array = a1
    saved_array2 = a2
    saved_array3 = a3

    locator = matplotlib.ticker.MultipleLocator(tick_freq)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(title, fontsize=26)
    plt.xlabel("iterations", fontsize=20)
    plt.ylabel("uncertainty", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    x = np.arange(max_iter)

    plt.plot(x, saved_array[:max_iter], label="N=30")
    plt.plot(x, saved_array2[:max_iter], label="N=12")
    plt.plot(x, saved_array3[:max_iter], label="N=5")
    plt.legend(loc='lower right', fontsize=18)
    plt.axis([0, max_iter, min_u, max_u])
    plt.grid(visible=True, alpha=0.3)

    plt.savefig(fname=f"graph_data/{filename}", dpi=400)
    plt.show()


def get_mean_from_folder(folder_name, lower, upper):
    file_paths = os.listdir(folder_name)
    file_arrs = []
    for path in file_paths:
        if path.endswith('.npy'):
            # print(folder_name + '/' + path)
            file_arrs.append(np.load(folder_name + '/' + path))

    file_arrs = np.array(file_arrs)
    file_arrs = np.sort(file_arrs, axis=0)

    mean = np.mean(file_arrs, axis=0)

    lower_error = file_arrs[lower]
    upper_error = file_arrs[upper]

    return mean, lower_error, upper_error

def plot_graph_with_error(a1, le1, ue1, a2, le2, ue2, a3, le3, ue3,
                          tick_freq, err_freq, max_iter, min_u, max_u, title, filename, labels):
    y1 = a1[:max_iter]
    y2 = a2[:max_iter]
    y3 = a3[:max_iter]

    err1 = (y1 - le1[:max_iter], ue1[:max_iter] - y1)
    err2 = (y2 - le2[:max_iter], ue2[:max_iter] - y2)
    err3 = (y3 - le3[:max_iter], ue3[:max_iter] - y3)

    locator = matplotlib.ticker.MultipleLocator(tick_freq)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(title, fontsize=26)
    plt.xlabel("iterations", fontsize=20)
    plt.ylabel("uncertainty", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    x = np.arange(max_iter)

    # plt.plot(x, y1[:max_iter], label="N=30")
    # plt.plot(x, y2[:max_iter], label="N=12")
    # plt.plot(x, y3[:max_iter], label="N=5")
    plt.errorbar(x, y1, err1, label=labels[0], errorevery=err_freq, mew=1, capsize=18)
    plt.errorbar(x, y2, err2, label=labels[1], errorevery=err_freq, mew=1, capsize=15)
    plt.errorbar(x, y3, err3, label=labels[2], errorevery=err_freq, mew=1, capsize=12)
    plt.legend(loc='lower right', fontsize=18)
    plt.axis([0, max_iter, min_u, max_u])
    plt.grid(visible=True, alpha=0.3)

    plt.savefig(fname=f"graph_data/{filename}", dpi=400)
    plt.show()


def plot_pso():
    folder_name_1 = 'evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it250_p30'
    mean_1, lower_error_1, upper_error_1 = get_mean_from_folder(folder_name_1, 5, 17)

    folder_name_2 = 'evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it50_p12'
    mean_2, lower_error_2, upper_error_2 = get_mean_from_folder(folder_name_2, 5, 17)

    folder_name_3 = 'evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it50_p5'
    mean_3, lower_error_3, upper_error_3 = get_mean_from_folder(folder_name_3, 5, 17)

    labels = ("N=30", "N=12", "N=5")

    tick_freq = 2
    err_freq = 4

    max_iter = 20
    min_u = 0
    max_u = 1700

    title = 'Particle swarm optimization'
    filename = "pso_graph_with_ci"

    plot_graph_with_error(mean_1, lower_error_1, upper_error_1,
                          mean_2, lower_error_2, upper_error_2,
                          mean_3, lower_error_3, upper_error_3,
                          tick_freq, err_freq, max_iter, min_u, max_u, title, filename, labels)


def plot_gd():
    folder_name_1 = 'evaluation_data/gd/gd_pss0.05_nss1e-05_it250_inm30'
    mean_1, lower_error_1, upper_error_1 = get_mean_from_folder(folder_name_1, 5, 17)

    folder_name_2 = 'evaluation_data/gd/gd_pss0.05_nss1e-05_it250_inm12'
    mean_2, lower_error_2, upper_error_2 = get_mean_from_folder(folder_name_2, 5, 17)

    folder_name_3 = 'evaluation_data/gd/gd_pss0.05_nss1e-05_it250_inm5'
    mean_3, lower_error_3, upper_error_3 = get_mean_from_folder(folder_name_3, 5, 17)

    labels = ("N=30", "N=12", "N=5")

    tick_freq = 25
    err_freq = 50

    max_iter = 250
    min_u = 0
    max_u = 1700

    title = 'Gradient descent'
    filename = "gd_graph"


    plot_graph_with_error(mean_1, lower_error_1, upper_error_1,
                          mean_2, lower_error_2, upper_error_2,
                          mean_3, lower_error_3, upper_error_3,
                          tick_freq, err_freq, max_iter, min_u, max_u, title, filename, labels)


def plot_lbfgs():
    folder_name_1 = 'evaluation_data/lbfgs/lbfgs_it250_inm30'
    mean_1, lower_error_1, upper_error_1 = get_mean_from_folder(folder_name_1, 5, 17)

    folder_name_2 = 'evaluation_data/lbfgs/lbfgs_it250_inm12'
    mean_2, lower_error_2, upper_error_2 = get_mean_from_folder(folder_name_2, 5, 17)

    folder_name_3 = 'evaluation_data/lbfgs/lbfgs_it250_inm5'
    mean_3, lower_error_3, upper_error_3 = get_mean_from_folder(folder_name_3, 5, 17)

    labels = ("N=30", "N=12", "N=5")

    tick_freq = 2
    err_freq = 4

    max_iter = 20
    min_u = 0
    max_u = 1700

    title = 'L-BFGS-B'
    filename = 'lbfgs_graph'


    plot_graph_with_error(mean_1, lower_error_1, upper_error_1,
                          mean_2, lower_error_2, upper_error_2,
                          mean_3, lower_error_3, upper_error_3,
                          tick_freq, err_freq, max_iter, min_u, max_u, title, filename, labels)


def plot_iter_comparison():
    folder_name_1 = 'evaluation_data/lbfgs/lbfgs_it250_inm30'
    mean_1, lower_error_1, upper_error_1 = get_mean_from_folder(folder_name_1, 5, 17)

    folder_name_2 = 'evaluation_data/gd/gd_pss0.05_nss1e-05_it250_inm30'
    mean_2, lower_error_2, upper_error_2 = get_mean_from_folder(folder_name_2, 5, 17)

    folder_name_3 = 'evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it250_p30'
    mean_3, lower_error_3, upper_error_3 = get_mean_from_folder(folder_name_3, 5, 17)

    labels = ("L-BFGS-B", "Gradient descent", "Particle swarm optimization")

    tick_freq = 25
    err_freq = 50

    max_iter = 250
    min_u = 0
    max_u = 1700

    title = 'Comparison of algorithms'
    filename = 'comparison_graph'



    plot_graph_with_error(mean_1, lower_error_1, upper_error_1,
                          mean_2, lower_error_2, upper_error_2,
                          mean_3, lower_error_3, upper_error_3,
                          tick_freq, err_freq, max_iter, min_u, max_u, title, filename, labels)




if __name__ == "__main__":
    plot_pso()
    plot_gd()
    plot_lbfgs()
    plot_iter_comparison()
