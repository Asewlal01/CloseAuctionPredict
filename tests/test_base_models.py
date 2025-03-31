from Models.ConvolutionalModels.BaseConvolve import BaseConvolve
from Models.ReccurentModels.BaseLSTM import BaseLSTM
from Models.TransformerModels.BaseTransformer import BaseTransformer
from Models.LinearRegression.BaseLinearRegression import BaseLinearRegression
import torch

# Global Variables
samples = 100
sequence_size = 20
features = 5
fc_neurons = [10, 20]
fc_dropout = 0.2

# Creation of samples
X_input = torch.rand(samples, sequence_size, features)
# X with one feature and one sample
X_singular = X_input[0, :, -1:]
# X with one feature and multiple samples
X_singular_samples = X_input[:, :, -1:]
# X with multiple features and one sample
X_singular_features = X_input[0, :, :]

# Z is a tensor with 2 features which represent overnight and yesterday's closing return
z_input = torch.rand(samples, 2)
z_singular = z_input[:1]

def test_base_logistic_regression():
    # Testing with one feature
    model = BaseLinearRegression(1, sequence_size)
    assert model(X_singular, z_singular).shape == (1, 1)
    assert model(X_singular_samples, z_input).shape == (samples, 1)

    # Testing with multiple features
    model = BaseLinearRegression(features, sequence_size)
    assert model(X_singular_features, z_singular).shape == (1, 1)
    assert model(X_input, z_input).shape == (samples, 1)

def test_base_convolve():
    # Convolutional Model settings
    conv_channels = [10, 10]
    kernel_size = 2

    # Testing with one feature
    model = BaseConvolve(1, sequence_size, conv_channels, fc_neurons, kernel_size)
    assert model(X_singular, z_singular).shape == (1, 1)
    assert model(X_singular_samples, z_input).shape == (samples, 1)

    # Testing with multiple features
    model = BaseConvolve(features, sequence_size, conv_channels, fc_neurons, kernel_size)
    assert model(X_singular_features, z_singular).shape == (1, 1)
    assert model(X_input, z_input).shape == (samples, 1)

def test_base_lstm():
    # LSTM Model settings
    hidden_size = 10
    lstm_size = 5

    # Testing with one feature
    model = BaseLSTM(1, hidden_size, lstm_size, fc_neurons)
    assert model(X_singular, z_singular).shape == (1, 1)
    assert model(X_singular_samples, z_input).shape == (samples, 1)

    # Testing with multiple features
    model = BaseLSTM(features, hidden_size, lstm_size, fc_neurons)
    assert model(X_singular_features, z_singular).shape == (1, 1)
    assert model(X_input, z_input).shape == (samples, 1)

def test_base_transformer():
    # Transformer Model settings
    embedding_size = 64
    num_heads = 4
    dropout = 0.1
    dim_feedforward = 256
    num_layers = 3

    # Testing with one feature
    model = BaseTransformer(1, sequence_size, embedding_size, num_heads, dropout, dim_feedforward,
                            num_layers, fc_neurons)
    assert model(X_singular, z_singular).shape == (1, 1)
    assert model(X_singular_samples, z_input).shape == (samples, 1)

    # Testing with multiple features
    model = BaseTransformer(features, sequence_size, embedding_size, num_heads, dropout, dim_feedforward,
                            num_layers, fc_neurons)
    assert model(X_singular_features, z_singular).shape == (1, 1)
    assert model(X_input, z_input).shape == (samples, 1)