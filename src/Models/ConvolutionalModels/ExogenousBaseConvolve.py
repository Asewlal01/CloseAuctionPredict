from Models.ExogenousBaseModel import ExogenousBaseModel
from Models.ConvolutionalModels.BaseConvolve import BaseConvolve

class ExogenousBaseConvolve(ExogenousBaseModel, BaseConvolve):
    """
    Base class for all the convolutional Models that use exogenous features. It inherits from the BaseModel class and
    the BaseConvolve class.
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
        ExogenousBaseModel.__init__(self, expected_dims)

    def build_model(self):
        """
        Build the model.
        """
        BaseConvolve.build_model(self)
