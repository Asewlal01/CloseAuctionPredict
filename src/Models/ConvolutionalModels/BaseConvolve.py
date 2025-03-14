from Models.BaseModel import BaseModel
from torch import nn
from Layers.PermuteLayer import PermuteLayer


class BaseConvolve(BaseModel):
    """
    Base class for all the convolutional Models. It inherits from the BaseModel class.
    """

    def __init__(self, feature_size: int, sequence_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, dropout: float=0.5):
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        expected_dims = 3
        super(BaseConvolve, self).__init__(expected_dims, dropout)

    def build_model(self) -> None:

        # Making sure that conv_channels and fc_neurons are lists
        self.conv_channels = [self.conv_channels] if type(self.conv_channels) == int else self.conv_channels
        self.fc_neurons = [self.fc_neurons] if type(self.fc_neurons) == int else self.fc_neurons

        # This layer permutes the dimensions of the input tensor by swapping 2nd and 3rd dimensions
        self.layers.append(PermuteLayer((0, 2, 1)))

        # Convolutional Layers
        in_channels = self.feature_size
        output_size = self.sequence_size
        for out_channels in self.conv_channels:
            self.layers.append(
                nn.Conv1d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, self.dilation)
            )

            # Size reduction from convolution
            output_size = (output_size + 2 * self.padding - self.kernel_size) // self.stride + 1

            # ReLU Activation
            self.layers.append(
                nn.ReLU()
            )

            # Max Pooling
            self.layers.append(
                nn.MaxPool1d(2)
            )

            # Size reduction from pooling
            output_size = (output_size - 2) // 2 + 1

            # Last output size is the number of channels
            in_channels = out_channels

            # If output_size is too small raise error
            if output_size < 1:
                raise ValueError("Input size is too small for the given convolutional layers")

        # Compute the size of the output of the convolutional self.layers
        sequence_size = output_size * in_channels

        # Flatten the output of the convolutional self.layers
        self.layers.append(nn.Flatten())

        # Fully Connected self.layers
        for out_neurons in self.fc_neurons:
            # Fully Connected Layer
            self.layers.append(
                nn.Linear(sequence_size, out_neurons)
            )

            # ReLU Activation
            self.layers.append(
                nn.ReLU()
            )
            sequence_size = out_neurons

        self.output_dim = sequence_size