from Models.ConvolutionalModels.BaseConvolve import BaseConvolve
from Models.ReccurentModels.BaseLSTM import BaseLSTM
from Models.TransformerModels.BaseTransformer import BaseTransformer
from Models.LogisticRegression.BaseLogisticRegression import BaseLogisticRegression
import torch

# Global Variables
samples = 100
sequence_size = 20
features = 5
fc_neurons = [10, 20]

def test_base_logistic_regression():
    # Testing with one feature
    model = BaseLogisticRegression(1, sequence_size)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, 1)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, 1)
    assert model(X).shape == (1, 1)

    # Testing with multiple features
    model = BaseLogisticRegression(features, sequence_size)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, features)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, features)
    assert model(X).shape == (1, 1)

def test_base_convolve():
    # Convolutional Model settings
    conv_channels = [10, 10]
    kernel_size = 2

    # Testing with one feature
    model = BaseConvolve(1, sequence_size, conv_channels, fc_neurons, kernel_size)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, 1)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, 1)
    assert model(X).shape == (1, 1)

    # Testing with multiple features
    model = BaseConvolve(features, sequence_size, conv_channels, fc_neurons, kernel_size)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, features)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, features)
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

    # Multiple Samples
    X = torch.rand(samples, sequence_size, 1)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, 1)
    assert model(X).shape == (1, 1)

    # Testing with multiple features
    model = BaseTransformer(features, sequence_size, embedding_size, num_heads, dropout, dim_feedforward,
                            num_layers, fc_neurons)

    # Multiple Samples
    X = torch.rand(samples, sequence_size, features)
    assert model(X).shape == (samples, 1)

    # Single Sample
    X = torch.rand(sequence_size, features)
    assert model(X).shape == (1, 1)