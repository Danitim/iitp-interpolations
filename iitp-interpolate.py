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
    "spline": spline_interpolation,
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
    "--showcase",
    "showcase",
    is_flag=True,
    default=False,
    help="Run all interpolation methods and compare visually.",
)
@click.option(
    "--save",
    "save_path",
    type=click.Path(),
    default=None,
    help="Path to save the interpolated image",
)
def main(image_path, x_scale, y_scale, method, showcase, save_path):
    """Image interpolation CLI. IMAGE_PATH is the input image."""
    image = Image.open(image_path)
    image_arr = np.asarray(image)

    new_height = int(x_scale * image_arr.shape[0])
    new_width = int(y_scale * image_arr.shape[1])
    
    if showcase:
        _showcase_all_methods(image_arr, new_height, new_width)
        return

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
    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(interpolated)
    ax[1].set_title("Interpolated")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def _showcase_all_methods(image_arr, new_h, new_w):
    print("[INFO] Running showcase mode...")

    results = {"Original": image_arr}
    for name, func in INTERPOLATION_METHODS.items():
        print(f"[INFO] Interpolating using {name}...")
        result = func(image_arr, new_h, new_w)
        results[name.capitalize()] = result

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Interpolation Showcase", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
