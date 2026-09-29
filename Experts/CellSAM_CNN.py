import torch
import torch.nn as nn

from Base_CNN import SimpleCNNEncoder


class CellSAMCNN(nn.Module):
    """
    CNN for extracting feature embeddings from CellSAM segmentation masks. 

    The network accepts CellSAM instance masks corresponding to different degradation levels of the same pathology image. 

    For the initial architecture, CellSAM instance labels are converted into binary masks:

        0 = background
        1 = detected cell
    
    The binary masks are then passed through a CNN encoder to produce fixed-size feature embeddings. 
    """

    def __init__(
        self,
        embedding_dim=128
    ):

        """
        Initialize the CellSAM CNN.

        Parameters
        ----------
        embedding_dim : int, default = 128
            Size of the feature embedding produced for each CellSAM mask.
        """

        super().__init__()

        self.encoder = SimpleCNNEncoder(
            input_channels=1,
            embedding_dim=embedding_dim
        )

    def forward(self, cellsam_masks):
        """
        Extract feature embeddings from CellSAM masks.

        Parameters
        ----------
        cellsam_masks : torch.Tensor
            Batch of CellSAM instance masks with shape:

                [batch_size, 1, height, width]

            For one pathology sample, the batch can contain:

                0 = clean
                1 = mild degradation
                2 = moderate degradation
                3 = severe degradation

        Returns
        -------
        torch.Tensor
            CellSAM feature embeddings with shape:

                [batch_size, embedding_dim]

        """

        binary_masks = (
            cellsam_masks > 0
        ).float()

        embeddings = self.encoder(
            binary_masks
        )

        return embeddings