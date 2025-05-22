from cnn import run_training
from Models.TransformerModels.ExogenousTradeLobTransformer import ExogenousTradeLobTransformer

def get_model(sequence_size):
    model = ExogenousTradeLobTransformer(
        sequence_size=sequence_size,
        embedding_size=32,
        num_heads=4,
        dim_feedforward=128,
        num_layers=2,
        fc_neurons = [128, 64],
        dropout = 0.2,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    sequence_size = 120
    model = get_model(sequence_size)
    epochs = 100
    lr = 1e-4
    name = 'transformer'
    run_training(model, epochs, lr, sequence_size, name)
