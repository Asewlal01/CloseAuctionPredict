from Models.BaseModel import BaseModel
from torch import nn
from Layers.PermuteLayer import PermuteLayer


class BaseConvolve(BaseModel):
    """
    Base class for all the convolutional Models. It inherits from the BaseModel class.
    """

    def __init__(self, feature_size: int, sequence_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: list[int], stride: int=1, padding: int=0, dilation: int=1, dropout: float=0):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock.

        Parameters
        ----------
        feature_size : Number of features in each channel of the input data.
        which will lead to new channels having 1 feature each.
        sequence_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers. Assumed to be constant for all layers.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        dropout : Dropout rate to use in the fully connected
        """

        self.feature_size = feature_size
        self.sequence_size = sequence_size
        self.conv_channels = conv_channels
        self.fc_neurons = fc_neurons
        self.kernel_sizes = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout

        expected_dims = 3
        super().__init__(expected_dims)

    def build_model(self) -> None:
        # Converting to lists
        if isinstance(self.conv_channels, int):
            self.conv_channels = [self.conv_channels]
        if isinstance(self.kernel_sizes, int):
            self.kernel_sizes = [self.kernel_sizes]
        if isinstance(self.fc_neurons, int):
            self.fc_neurons = [self.fc_neurons]

        # This layer permutes the dimensions of the input tensor by swapping 2nd and 3rd dimensions
        self.layers.append(PermuteLayer((0, 2, 1)))

        # Convolutional Layers
        in_channels = self.feature_size
        output_size = self.sequence_size
        for out_channels, kernel_size in zip(self.conv_channels, self.kernel_sizes):
            self.layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, self.stride, self.padding, self.dilation)
            )

            # Size reduction from convolution
            output_size = (output_size + 2 * self.padding - kernel_size) // self.stride + 1

            # ReLU Activation
            self.layers.append(
                nn.ReLU()
            )

            # Last output size is the number of channels
            in_channels = out_channels

            # If output_size is too small raise error
            if output_size < 1:
                raise ValueError("Input size is too small for the given convolutional layers")

        # Compute the size of the output of the convolutional self.layers
        sequence_size = output_size * in_channels

        # Flatten the output of the convolutional self.layers
        self.layers.append(nn.Flatten())

        # Dropout Layer
        self.layers.append(nn.Dropout(self.dropout))

        self.output_dim = sequence_size