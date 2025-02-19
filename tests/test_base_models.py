from models.ConvolutionalModels.BaseConvolve import BaseConvolve
from models.ReccurentModels.BaseLSTM import BaseLSTM
import torch

# Global Variables
samples = 100
sequence_size = 20
features = 5
fc_neurons = [10, 20]

def test_base_convolve():
    # Convolutional Model settings
    conv_channels = [10, 10]
    kernel_size = 2

    # Testing with one feature
    model = BaseConvolve(1, sequence_size, conv_channels, fc_neurons, kernel_size)

    # Multiple Samples
    X = torch.rand(samples, 2, sequence_size)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(2, sequence_size)
    assert model(X).shape == (1, 1)

    # Testing with multiple features
    model = BaseConvolve(features, sequence_size, conv_channels, fc_neurons, kernel_size)

    # Multiple Samples
    X = torch.rand(samples, 2, sequence_size, features)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(2, sequence_size, features)
    assert model(X).shape == (1, 1)

def test_base_lstm():
    # LSTM Model settings
    hidden_size = 10
    lstm_size = 5

    # Testing with one feature
    model = BaseLSTM(1, hidden_size, lstm_size, fc_neurons)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, 1)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, 1)
    assert model(X).shape == (1, 1)

    # Testing with multiple features
    model = BaseLSTM(features, hidden_size, lstm_size, fc_neurons)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, features)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, features)
    assert model(X).shape == (1, 1)

