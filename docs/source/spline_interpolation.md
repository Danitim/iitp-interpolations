# Spline Interpolation

## Overview

Spline interpolation is a smooth and flexible resampling method that uses cubic splines to estimate pixel values. It is particularly effective for resizing images while preserving smooth gradients and minimizing artifacts.

## How It Works

1. **Input Image**: Provide a 2D (grayscale) or 3D (RGB) image.
2. **Cubic Kernel**: Use a cubic spline kernel to calculate weights for surrounding pixels.
3. **Weighted Average**: Compute the weighted average of the surrounding pixels for each target pixel.
4. **Output Image**: Return the resized image.

## Usage

To use the spline interpolation method, follow these steps:

1. Import the `spline_interpolation` function.
2. Provide the input image and desired dimensions.
3. Call the function to get the resized image.

## Example

```python
import numpy as np
from methods.spline import spline_interpolation

# Example input image (grayscale)
image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

# Resize to 4x4 using spline interpolation
resized_image = spline_interpolation(image, new_height=4, new_width=4)

print("Resized Image:")
print(resized_image)
```

## Advantages

- Produces smooth and visually appealing results.
- Effective for both upscaling and downscaling images.
- Minimizes artifacts compared to simpler methods like bilinear interpolation.

## Limitations

- Computationally expensive compared to bilinear interpolation.
- May introduce slight overshooting or ringing artifacts in some cases.

## Notes

- The method supports both grayscale and RGB images.
- Ensure the input image is a valid NumPy array with 2D or 3D dimensions.
