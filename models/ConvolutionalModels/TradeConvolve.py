from models.ConvolutionalModels.BaseConvolve import BaseConvolve

class TradeConvolve(BaseConvolve):
    """
    Convolutional Neural Network for predicting the Closing Price of a stock using the Trade data. This model assumes
    that the trade data has prices and volumes, which are given in their own channels.
    """
    def __init__(self, input_size: int, conv_channels: list[int], fc_neurons: list[int],
                 kernel_size: int, stride: int=1, padding: int=0, dilation: int=1):
        """
        Initializes the Convolutional Neural Network for predicting the Closing Price of a stock using the Trade
        data.

        Parameters
        ----------
        input_size : Number of sequence steps in the input data
        conv_channels : Number of channels after each convolutional layer
        fc_neurons : Number of neurons in each fully connected layer
        kernel_size : Size of the kernel for the convolutional layers. Assumed to be constant for all layers.
        stride : Stride for the convolutional layers. Assumed to be constant for all layers
        padding : Padding for the convolutional layers. Assumed to be constant for all layers
        dilation : Dilation for the convolutional layers. Assumed to be constant for all layers
        """

        super(TradeConvolve, self).__init__(1, input_size, conv_channels, fc_neurons,
                                            kernel_size, stride, padding, dilation)