import torch
import torch.nn as nn

class ShallowDNN(nn.Module):
    """
    Shallow neural network for combining RGB and CellSAM feature embeddings and producing final predictions. 

    RGB and CellSAM embeddings are concatenated directly along the feature dimension before being passed through the network.
    """

    def __init__(
        self,
        rgb_embedding_dim=128,
        cellsam_embedding_dim=128,
        hidden_dim=128,
        num_classes=2
    ):

        """
        Initialize the shallow DNN.

        Parameters
        ----------
        rgb_embedding_dim : int, default=128
            Size of the RGB CNN embedding.

        cellsam_embedding_dim : int, default=128
            Size of the CellSAM CNN embedding.

        hidden_dim : int, default=128
            Number of features in the hidden layer.

        num_classes : int, default=2
            Number of output classes.
        """

        super().__init__()

        combined_dim = (
            rgb_embedding_dim + cellsam_embedding_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(
                combined_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        rgb_embedding, 
        cellsam_embedding
    ):

        """
        Concatenate RGB and CellSAM embedding and produce predictions.

        Parameters
        ----------
        rgb_embedding : torch.Tensor
            RGB CNN embeddings with shape:

                [batch_size, rgb_embedding_dim]
            
        cellsam_embedding : torch.Tensor
            CellSAM CNN embeddings with shape:

                [batch_size, cellsam_embedding_dim]

        Returns
        -------
        torch.Tensor
            Classification logits with shape:

                [batch_size, num_classes]
        """

        combined_embedding = torch.cat(
            (
                rgb_embedding, 
                cellsam_embedding
            ),
            dim=1
        )

        output = self.classifier(
            combined_embedding
        )

        return output