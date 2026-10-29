import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

class SimpleCNNEncoder(nn.Module):
    """
    Simple convolutional neural network for extracting features from pathology images.

    The network accepts image tensors and converts them into a fixed-size feature embedding. This embedding can 
    later be concatenated with embeddings from other network branches before being passed to a downstream 
    classifier. 
    """

    def __init__(
        self, input_channels=3, embedding_dim=128
    ):
        """
        Initialize the CNN encoder.

        Parameters
        ----------
        input_channels : int, default=3
            Number of channels in the input image.
            Standard RGB pathology images use 3 channels. 

        embedding_dim : int, default = 128
            Size of the feature embedding produced by the network.
        """

        super().__init__()

        #Convolutional feature extractor
        self.features = nn.Sequential(

            nn.Conv2d(
                input_channels, 32, kernel_size=3, padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(32, 64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2
            ),
            
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1,1)
            )
        )

        self.embedding = nn.Linear(128, embedding_dim
        )
    def forward(self, image):
        """
        Extract a feature embedding from a batch of images. 

        Parameters
        ----------
        image : torch.Tensor
            Image tensor with shape:

                [batch_size, channels, height, width]

        Returns
        -------
        torch.Tensor
            Feature embedding with shape: 
                
                [batch_size, embedding_dim]

        """

        features = self.features(image)

        features = torch.flatten(
            features,
            start_dim=1
        )

        embedding = self.embedding(
            features
        )

        return embedding
    
if __name__ == "__main__":

    model = SimpleCNNEncoder()

    transform = transforms.Compose([
        transforms.Resize(
            (224,224)
        ),
        transforms.ToTensor()
    ])

    image = Image.open(
        r"C:\Users\ziaoo\Downloads\C1102-23-1.tif_lux_13817_luy_18392_h_216_w_156.png"
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    embedding = model(
        image_tensor
    )

    print(
        "Input shape:",
        image_tensor.shape
    )

    print(
        "Embedding shape:",
        embedding.shape
    )