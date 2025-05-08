# Lanczos Interpolation

## Overview

Lanczos interpolation is a high-quality resampling method that uses a sinc-based kernel to compute the weighted average of surrounding pixels. It is particularly effective for resizing images with minimal artifacts.

## How It Works

1. **Input Image**: Provide a 2D (grayscale) or 3D (RGB) image.
2. **Kernel Calculation**: Compute the Lanczos kernel based on the distance from the interpolation center.
3. **Weighted Average**: Apply the kernel to surrounding pixels to compute the new pixel value.
4. **Output Image**: Return the resized image.

## Usage

To use the Lanczos interpolation method, follow these steps:

1. Import the `lanczos_interpolation` function.
2. Provide the input image, desired dimensions, and kernel size.
3. Call the function to get the resized image.

## Example

```python
import numpy as np
from methods.lanczos import lanczos_interpolation

# Example input image (grayscale)
image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

# Resize to 4x4 using Lanczos interpolation
resized_image = lanczos_interpolation(image, new_height=4, new_width=4, a=3)

print("Resized Image:")
print(resized_image)
```

## Advantages

- Produces high-quality results with minimal artifacts.
- Effective for both upscaling and downscaling images.

## Limitations

- Computationally expensive compared to simpler methods like bilinear interpolation.
- Kernel size (`a`) affects performance and quality; larger values increase computation time.

## Notes

- The method supports both grayscale and RGB images.
- Ensure the input image is a valid NumPy array with 2D or 3D dimensions.
- The default kernel size (`a=3`) works well for most use cases.
