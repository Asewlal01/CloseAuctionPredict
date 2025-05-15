from cnn import run_training
from Models.TransformerModels.LobTransformer import LobTransformer

def get_model(sequence_size):
    model = LobTransformer(
        sequence_size=sequence_size,
        embedding_size=16 * 2,
        num_heads=2**2,
        dim_feedforward=16 * 2 * 6,
        num_layers=4,
        fc_neurons = [8*3, 8*8, 8*8, 8*1, 8*8],
        dropout = 0,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    sequence_size = 8 * 8
    model = get_model(sequence_size)
    epochs = 6 * 5
    lr = 1e-3
    name = 'transformer'
    run_training(model, epochs, lr, sequence_size, name)
