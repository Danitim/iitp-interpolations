import numpy as np
import matplotlib.pyplot as plt

from methods import node_wise_polynomial_interpolation

if __name__ == "__main__":
    
    # generate data
    x = np.linspace(0, 2*np.pi, 10)
    y = np.exp(x)

    # interpolate
    x_new = np.random.rand(5)*2*np.pi
    f = node_wise_polynomial_interpolation(x, y, x_new, degree=3)

    # plot
    plt.plot(x, y, 'o', label='data')
    plt.plot(x_new, f, 'x', label='interpolation')
    plt.legend()
    plt.show()