import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.methods import bilinear_interpolation


def demo():
    image = Image.open("examples/succ.png")
    image_arr = np.asarray(image)

    # interpolate
    ratio = 0.5
    x_new = int(ratio * image_arr.shape[0])
    y_new = int(ratio * image_arr.shape[1])
    interpolated_image_arr = bilinear_interpolation(image_arr, x_new, y_new)
    print(f"Original image shape: {image_arr.shape}")
    print(f"Interpolated image shape: {interpolated_image_arr.shape}")

    # show image before and after
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_arr, cmap="gray")
    ax[0].set_title("Original image")
    ax[1].imshow(interpolated_image_arr, cmap="gray")
    ax[1].set_title("Interpolated image")
    plt.show()
    return
