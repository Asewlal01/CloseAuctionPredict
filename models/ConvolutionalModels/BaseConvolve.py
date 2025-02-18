from models.BaseModel import BaseModel
from torch import nn


class BaseConvolve(BaseModel):
    """
    Base class for all the convolutional models. It inherits from the BaseModel class.
    """

    def __init__(self, feature_size, input_size, conv_channels, fc_neurons, kernel_size, stride=1, padding=0,
                 dilation=1):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock using the Trade data.

        Parameters
        ----------
        feature_size : Number of features in each channel of the input data. If the features is greater than 1, then
        the first convolutional layer will apply a 2D convolution with a kernel size of (kernel_size, feature_size)
        which will lead to new channels having 1 feature each.
        input_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers. Assumed to be constant for all layers.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        """
        super(BaseConvolve, self).__init__()

        # Instantiate the layers
        layers = []

        # Making sure that conv_channels and fc_neurons are lists
        conv_channels = [conv_channels] if type(conv_channels) == int else conv_channels
        fc_neurons = [fc_neurons] if type(fc_neurons) == int else fc_neurons

        # Convolutional Layers
        in_channels = 2
        output_size = input_size
        for out_channels in conv_channels:
            # Go for 2D convolution if there are more than 1 feature
            if feature_size > 1:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, (kernel_size, feature_size), stride, padding, dilation)
                )

                # Remove last dimension of the output
                layers.append(
                    nn.Flatten(2)
                )

                feature_size = 1
            else:
                layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
                )

            # Size reduction from convolution
            output_size = (output_size + 2 * padding - kernel_size) // stride + 1

            # ReLU Activation
            layers.append(
                nn.ReLU()
            )

            # Max Pooling
            layers.append(
                nn.MaxPool1d(2)
            )

            # Size reduction from pooling
            output_size = (output_size - 2) // 2 + 1

            # Last output size is the number of channels
            in_channels = out_channels

        # Compute the size of the output of the convolutional layers
        input_size = output_size * in_channels

        # Flatten the output of the convolutional layers
        layers.append(nn.Flatten())

        # Fully Connected Layers
        for out_neurons in fc_neurons:
            # Fully Connected Layer
            layers.append(
                nn.Linear(input_size, out_neurons)
            )

            # ReLU Activation
            layers.append(
                nn.ReLU()
            )
            input_size = out_neurons

        # Output Layer
        layers.append(
            nn.Linear(input_size, 1)
        )

        # Save the layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x