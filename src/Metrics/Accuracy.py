import torch.nn as nn

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, y_pred, y_true):
        pred_bool = (y_pred > 0).float()
        true_bool = (y_true > 0).float()
        accuracy = (pred_bool == true_bool).float()
        return accuracy.mean()