import torch.nn as nn

class TradeConvolve(nn.Module):
    def __init__(self, input_size: int, conv_size: list[int], fc_size: list[int],
                 kernel_size: int = 2, stride: int = 1, padding: int = 0):
        """
        Convolutional Neural Network for predicting the Closing Price of a stock using the Trade data. Input data
        is assumed to be a 2D tensor with the first dimension being the price and the second dimension being the volume.
        Those two variables act as channels for the convolutional layers. The output of the convolutional layers is the
        prediction of the closing price.

        Parameters
        ----------
        input_size : Number of time steps in the input data
        conv_size : Number of channels after each convolutional layer
        fc_size : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers
        stride : Stride for the convolutional layers
        padding : Padding for the convolutional layers
        """
        super(TradeConvolve, self).__init__()

        # Saving the parameters for the forward pass
        self.input_size = input_size
        self.conv_size = conv_size
        self.fc_size = fc_size

        # Saving the layers
        layers = []
        if type(conv_size) == int:
            print("conv_size should be a list of integers")
            conv_size = [conv_size]

        # Convolutional Layers
        in_channels = 2
        output_size = input_size
        for out_channels in conv_size:
            # Convolution
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
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
        layers.append(
            nn.Flatten()
        )

        # Fully Connected Layers
        if type(fc_size) == int:
            print("fc_size should be a list of integers")
            fc_size = [fc_size]
        for out_size in fc_size:
            # Fully Connected Layer
            layers.append(
                nn.Linear(input_size, out_size)
            )

            # ReLU Activation
            layers.append(
                nn.ReLU()
            )
            input_size = out_size

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