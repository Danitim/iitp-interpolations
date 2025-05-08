# Bilinear Interpolation

## Overview

Bilinear interpolation is a resampling method that uses the values of the four nearest pixels to estimate a new pixel value. It is commonly used for resizing images while maintaining smooth transitions between pixel values.

## How It Works

1. **Input Image**: Provide a 2D (grayscale) or 3D (RGB) image.
2. **Grid Mapping**: Map the coordinates of the target image to the source image.
3. **Interpolation**: Compute the weighted average of the four nearest pixels for each target pixel.
4. **Output Image**: Return the resized image.

## Usage

To use the bilinear interpolation method, follow these steps:

1. Import the `bilinear_interpolation` function.
2. Provide the input image and desired dimensions.
3. Call the function to get the resized image.

## Example

```python
import numpy as np
from methods.bilinear import bilinear_interpolation

# Example input image (grayscale)
image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

# Resize to 4x4
resized_image = bilinear_interpolation(image, new_height=4, new_width=4)

print("Resized Image:")
print(resized_image)
```

## Advantages

- Smooth transitions between pixels.
- Suitable for resizing images with minimal artifacts.

## Limitations

- Computationally more expensive than nearest-neighbor interpolation.
- May introduce slight blurring in the resized image.

## Notes

- The method supports both grayscale and RGB images.
- Ensure the input image is a valid NumPy array with 2D or 3D dimensions.
