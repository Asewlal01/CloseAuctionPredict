from Models.ConvolutionalModels.BaseConvolve import BaseConvolve

class LobConvolve(BaseConvolve):
    """
    Convolutional Neural Network for predicting the Stock Prices using Limit Order Book data. It is
    assumed that the LOB data has a price and volume for each price level, with 5 levels for each side of the book,
    leading to a total of 20 features.
    """
    def __init__(self, sequence_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: list[int], stride: int=1, padding: int=0, dilation: int=1, dropout: float=0):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock using the Trade
        data.

        Parameters
        ----------
        sequence_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for each convolutional layer.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        dropout : Dropout rate to use in the fully connected
        """

        feature_size = 20
        super().__init__(feature_size, sequence_size, conv_channels, fc_neurons,
                                            kernel_size, stride, padding, dilation, dropout)