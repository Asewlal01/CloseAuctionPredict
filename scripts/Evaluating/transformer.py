from Models.TransformerModels.LimitOrderBookTransformer import LimitOrderBookTransformer
from cnn import main

if __name__ == '__main__':
    model = LimitOrderBookTransformer(
        sequence_size=420,
        embedding_size=64,
        num_heads=4,
        dropout=0.1,
        dim_feedforward=32,
        num_layers=2,
        fc_neurons=[32, 16],
    )
    model.to('cuda')
    model_name = 'transformer'
    main(model, model_name)