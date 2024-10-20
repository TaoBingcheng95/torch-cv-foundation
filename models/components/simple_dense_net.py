import torch
from torch import nn

from torchvision.models.densenet import DenseNet
from segmentation_models_pytorch.encoders import densenet

from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SegmentationHead(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 784):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Assuming binary segmentation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """
        Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


class ModernDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        output_size: int = 10,
        segmentation_output_size: int = 784
    ) -> None:
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, hidden_size)
        self.classification_head = ClassificationHead(hidden_size, output_size)
        self.segmentation_head = SegmentationHead(hidden_size, segmentation_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.encoder(x)
        x = self.decoder(x)

        # Determine the task based on the shape of the input
        if x.size(1) == 784:  # Assume 784 indicates a segmentation task
            return self.segmentation_head(x)
        else:
            return self.classification_head(x)

if __name__ == "__main__":

    # model = SimpleDenseNet()
    model = ModernDenseNet()
    input_size = (1, 1, 28, 28)
    summary(model, input_size=input_size)

    # model.eval()
    # inputs = torch.randn(input_size)
    # output = model(inputs)
    # print(output.shape)


