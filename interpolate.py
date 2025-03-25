import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import click

from methods.bilinear import bilinear_interpolation


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("x_scale", type=float)
@click.argument("y_scale", type=float)
@click.option(
    "--save",
    "save_path",
    type=click.Path(),
    default=None,
    help="Path to save the interpolated image",
)
def main(image_path, x_scale, y_scale, save_path):
    image = Image.open(image_path)
    image_arr = np.asarray(image)

    # interpolate
    x_new, y_new = (
        int(x_scale * image_arr.shape[0]),
        int(y_scale * image_arr.shape[1]),
    )
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

    # save image if save_path is provided
    if save_path:
        interpolated_image = Image.fromarray(interpolated_image_arr)
        interpolated_image.save(save_path)
        print(f"Interpolated image saved to {save_path}")
    return


if __name__ == "__main__":
    main()
