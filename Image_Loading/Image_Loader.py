from pathlib import Path

import cv2
import numpy as np
import torch


class ImageLoader:
    """
    Load pathology images and convert them into PyTorch tensors.

    The loader supports both individual images and datasets containing multiple images. Images are converted from OpenCV's BGR format to RGB before being converted into PyTorch tensors. 

    Output tensors use channel-first format:

        [channels, height, width]

    which is the format expected by PyTorch convolutional neural networks
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    }

    def __init__(
        self,
        resize=None,
        interpolation=cv2.INTER_AREA
    ):
        """
        Initialize the image loader.

        Parameters
        ----------
        resize : tuple(int, int) or None, default=None
            Optional image size in the form:

                (width, height)
            If None, the original image dimensions are preserved.

        interpolation : int, default=cv2.INTER_AREA
            OpenCV interpolation method used when resizing images. 
        """

        self.resize = resize
        self.interpolation = interpolation

    def load(self, image_path):
        """
        Load one image and convert it into a PyTorch tensor.

        Parameters
        ----------
        image_path : str or pathlib.Path
            Path to the image.

        Returns
        -------
        torch.Tensor
            RGB image tensor with shape:

                [3, height, width]

            Pixel values are converted from uint8 [0, 255] to float32 [0, 1].

        """

        image_path = Path(image_path)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.resize is not None:
            image = cv2.resize(
                image,
                self.resize,
                interpolation=self.interpolation
            )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.transpose(
            image, (2, 0, 1)
        )

        tensor = torch.from_numpy(
            image
        )

        return tensor

    def load_dataset(self, directory):
        """
        Load all supported images contained in a directory.

        The directory is searched recursively. 

        Parameters
        ----------
        directory : str or pathlib.Path
            Root directory containing the image dataset.

        Returns
        -------
        list of tuple
            Each item contains:

            (image_path, image_tensor)

        where image_tensor has shape:

            [3, height, width]
        """

        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Dataset directory does not exist: {directory}"
            )

        dataset = []

        for image_path in directory.rglob("*"):

            if (image_path.suffix.lower()
            not in self.IMAGE_EXTENSIONS
            ):

                continue
            
            image_tensor = self.load(
                image_path
            )

            dataset.append(
                (
                    image_path,
                    image_tensor
                )
            )

        return dataset

if __name__ == "__main__":
    loader = ImageLoader(
        resize=(224,224)
    )

    image_tensor = loader.load(
        r"C:\Users\ziaoo\Downloads\C1102-23-1.tif_lux_13817_luy_18392_h_216_w_156.png"
    )

    print(
        "Image tensor shape:",
        image_tensor.shape
    )

    print(
        "Tensor dtype:",
        image_tensor.dtype
    )

    print(
        "Minimum pixel value:",
        image_tensor.min().item()
    )

    print(
        "Maximum pixel value:",
        image_tensor.max().item()
    )
