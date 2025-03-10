from Models.ConvolutionalModels.BaseConvolve import BaseConvolve

class TradeConvolve(BaseConvolve):
    """
    Convolutional Neural Network for predicting the Closing Price of a stock using the Trade data. This model assumes
    that the trade data has prices and volumes, which are given in their own channels.
    """
    def __init__(self, sequence_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, dropout: float=0.5):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock using the Trade
        data.

        Parameters
        ----------
        sequence_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers. Assumed to be constant for all layers.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        dropout : Dropout rate to use in the fully connected
        """

        # Trade data has 5 channels: Open, High, Low, Close, Volume
        feature_size = 5
        super(TradeConvolve, self).__init__(feature_size, sequence_size, conv_channels, fc_neurons,
                                            kernel_size, stride, padding, dilation, dropout)