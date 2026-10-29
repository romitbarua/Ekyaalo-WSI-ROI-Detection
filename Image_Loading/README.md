# IMAGE LOADING
Overview:
The Image_Loading directory contains utilities for loading pathology images and converting them into PyTorch-compatible tensors.

The Image_Loader.py module provides a reusable interface for loading individual images and recursively processing image datasets. 

The module handles image decoding, color conversion, optional resizing, pixel normalization, and tensor conversion. 

File Description:
Image_Loader.py:
Implements the ImageLoader class, which loads pathology images and converts them into PyTorch tensors suitable for convolutional neural networks. 

The class supports two primary operations: 
1. Loading individual pathology images
2. Recursively loading supported images from a dataset directory

# Image Preprocessing:
Each image undergoes the following preprocessing operations. 

Image loading:
Images are loaded using OpenCV (cv2.imread)

Supported image extensions include:
- .png 
- .jpg
- .jpeg
- .bmp
- .tif
- .tiff

Color conversion:
OpenCV loads color images in BGR format.

The loader converts images from BGR to RGB before tensor conversion.

Optional Resizing:
Images can optionally be resized to a specified spatial resolution. 

The resize parameter follows the format:
(width, height)

For the initial Ekyaalo architecture, images are typically resized to 224 x 224 pixels. 

The default resizing interpolation method is cv2.INTER_AREA.

If no resize dimensions are provided, the original image dimensions are preserved. 

Pixel Normalization:
Image pixel values are converted into 32-bit floating-point values and divided by 255. 

For standard 8-bit images, this transforms the pixel range from:
[0, 255] -> [0.0, 1.0]

No additional channel-wise mean or standard-deviation normalization is currently performed. 

Tensor Conversion:
Images are converted from the NumPy channel-last representation:
[height, width, channels]
to the Pytorch channel-first representation:
[channels, height, width]
This format is compatible with PyTorch convolutional neural networks.

For images resized to 224 x 224 pixels, the resulting tensor has shape [3, 224, 224].

# Usage example:
from Image_Loader import ImageLoader

loader = ImageLoader(resize=(224, 224))
image_tensor = loader.load("path/to/image.png")

print(image_tensor.shape)