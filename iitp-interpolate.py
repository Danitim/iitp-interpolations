import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import click

from methods.bilinear import bilinear_interpolation
from methods.lanczos import lanczos_interpolation
from methods.spline import spline_interpolation


INTERPOLATION_METHODS = {
    "bilinear": bilinear_interpolation,
    "lanczos": lanczos_interpolation,
    "spline": spline_interpolation
}


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("x_scale", type=float)
@click.argument("y_scale", type=float)
@click.option(
    "--method",
    "-m",
    type=click.Choice(INTERPOLATION_METHODS.keys(), case_sensitive=False),
    default="bilinear",
    show_default=True,
    help="Interpolation method to use",
)
@click.option(
    "--save",
    "save_path",
    type=click.Path(),
    default=None,
    help="Path to save the interpolated image",
)
def main(image_path, x_scale, y_scale, method, save_path):
    """Image interpolation CLI. IMAGE_PATH is the input image."""
    image = Image.open(image_path)
    image_arr = np.asarray(image)

    new_height = int(x_scale * image_arr.shape[0])
    new_width = int(y_scale * image_arr.shape[1])

    interpolation_func = INTERPOLATION_METHODS[method.lower()]

    interpolated = interpolation_func(image_arr, new_height, new_width)

    print(f"[INFO] Original shape:     {image_arr.shape}")
    print(f"[INFO] Interpolated shape: {interpolated.shape}")

    _show_images(image_arr, interpolated)

    if save_path:
        Image.fromarray(interpolated).save(save_path)
        print(f"[INFO] Saved interpolated image to '{save_path}'")


def _show_images(original, interpolated):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(interpolated, cmap="gray")
    ax[1].set_title("Interpolated")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()