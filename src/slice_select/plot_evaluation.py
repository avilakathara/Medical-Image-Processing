import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker



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


def plot_lbfgs():
    a1 = np.load('evaluation_data/lbfgs/lbfgs_it250_f8_ina4_inm30.npy')
    a2 = np.load('evaluation_data/lbfgs/lbfgs_it250_f8_ina4_inm12.npy')
    a3 = np.load('evaluation_data/lbfgs/lbfgs_it250_f8_ina4_inm5.npy')

    tick_freq = 2

    max_iter = 20
    min_u = 1000
    max_u = 1500

    title = 'L-BFGS-B'
    filename = 'lbfgs_graph'

    plot_graph(a1, a2, a3, tick_freq, max_iter, min_u, max_u, title, filename)


def plot_gd():
    a1 = np.load('evaluation_data/gd/gd_pss0.05_nss1e-05_it250_f8_ina4_inm30.npy')
    a2 = np.load('evaluation_data/gd/gd_pss0.05_nss1e-05_it250_f8_ina4_inm12.npy')
    a3 = np.load('evaluation_data/gd/gd_pss0.05_nss1e-05_it250_f8_ina4_inm5.npy')

    tick_freq = 25

    max_iter = 250
    min_u = 600
    max_u = 1500

    title = 'Gradient descent'
    filename = "gd_graph"

    plot_graph(a1, a2, a3, tick_freq, max_iter, min_u, max_u, title, filename)


def plot_pso():
    a1 = np.load('evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it250_f8_ina4_p30.npy')
    a2 = np.load('evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it250_f8_ina4_p12.npy')
    a3 = np.load('evaluation_data/pso/pso_o0.8_c1=0.2_c2=0.2_it250_f8_ina4_p5.npy')

    tick_freq = 2

    max_iter = 20
    min_u = 1000
    max_u = 1400

    title = 'Particle swarm optimization'
    filename = "pso_graph"

    plot_graph(a1, a2, a3, tick_freq, max_iter, min_u, max_u, title, filename)


if __name__ == "__main__":
    plot_pso()
    plot_gd()
    plot_lbfgs()
