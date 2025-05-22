from cnn import run_training
from Models.TransformerModels.LobTransformer import LobTransformer

def get_model(sequence_size):
    model = LobTransformer(
        sequence_size=sequence_size,
        embedding_size=32,
        num_heads=4,
        dim_feedforward=64,
        num_layers=4,
        fc_neurons = [128, 64],
        dropout = 0.1,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    sequence_size = 240
    model = get_model(sequence_size)
    epochs = 100
    lr = 1e-3
    name = 'transformer'
    run_training(model, epochs, lr, sequence_size, name)
