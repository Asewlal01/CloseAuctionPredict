import torch

class LastOutputLSTM(torch.nn.Module):
    """
    Layer designed to extract the output state from a LSTM layer. The LSTM layer returns (output, (h_n, c_n)) where
    only the output is needed for the task. More specifically, we only need the last output of the sequence.
    """
    def __init__(self):
        """
        Initializes the LastOutputLSTM layer
        """
        super(LastOutputLSTM, self).__init__()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return x[0][:, -1]