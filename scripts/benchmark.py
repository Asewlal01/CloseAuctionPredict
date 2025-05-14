from Models.ConvolutionalModels.LobConvolve import LobConvolve
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager
from Modeling.WalkForwardTester import WalkForwardTester

path = 'data/intraday'
train_manager = BaseDatasetManager(path, 1)
test_manager = BaseDatasetManager(path, 1)
train_manager.setup_dataset('2021-01')
test_manager.setup_dataset('2021-02')

model = LobConvolve(
    sequence_size=360,
    conv_channels=[128, 128, 128, 128, 128],
    fc_neurons=[128, 128, 128, 128, 128],
    kernel_size=[3, 3, 3, 3, 3],
    dropout=0.5
)
model.to('cuda')
tester = WalkForwardTester(model, train_manager, test_manager, 360)
epochs = 10
batch_size = 16
lr = 1e-3
tester.train(epochs, lr, batch_size, True)