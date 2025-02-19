from models.BaseModel import BaseModel
from torch import nn


class BaseConvolve(BaseModel):
    """
    Base class for all the convolutional models. It inherits from the BaseModel class.
    """

    def __init__(self, feature_size: int, sequence_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: int, stride: int=1, padding: int=0, dilation: int=1):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock.

        Parameters
        ----------
        feature_size : Number of features in each channel of the input data. If the features is greater than 1, then
        the first convolutional layer will apply a 2D convolution with a kernel size of (kernel_size, feature_size)
        which will lead to new channels having 1 feature each.
        sequence_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers. Assumed to be constant for all layers.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        """
        expected_dims = 3 + (feature_size > 1)
        super(BaseConvolve, self).__init__(expected_dims)

        # Making sure that conv_channels and fc_neurons are lists
        conv_channels = [conv_channels] if type(conv_channels) == int else conv_channels
        fc_neurons = [fc_neurons] if type(fc_neurons) == int else fc_neurons

        # Convolutional Layers
        in_channels = 2
        output_size = sequence_size
        for out_channels in conv_channels:
            # Go for 2D convolution if there are more than 1 feature
            if feature_size > 1:
                self.layers.append(
                    nn.Conv2d(in_channels, out_channels, (kernel_size, feature_size), stride, padding, dilation)
                )

                # Remove last dimension of the output
                self.layers.append(
                    nn.Flatten(2)
                )

                feature_size = 1
            else:
                self.layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
                )

            # Size reduction from convolution
            output_size = (output_size + 2 * padding - kernel_size) // stride + 1

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
        for out_neurons in fc_neurons:
            # Fully Connected Layer
            self.layers.append(
                nn.Linear(sequence_size, out_neurons)
            )

            # ReLU Activation
            self.layers.append(
                nn.ReLU()
            )
            sequence_size = out_neurons

        # Output Layer
        self.layers.append(
            nn.Linear(sequence_size, 1)
        )

        # Save the layers
        self.layers = nn.ModuleList(self.layers)