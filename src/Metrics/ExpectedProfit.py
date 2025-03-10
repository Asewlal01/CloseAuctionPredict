import torch.nn as nn
import torch

class ExpectedProfit(nn.Module):
    """
    This metric computes the expected profit of a trading strategy. The optimal threshold can be optimized using during
    training.
    """
    def __init__(self, k: int=50, init_tau: float=0.5, requires_tau_grad: bool=True, lr: float=1e-4):
        """
        Initialize the ExpectedProfit object.

        Parameters
        ----------
        k : Value that handles the smoothness of the sigmoid function
        init_tau : Initial value of the threshold
        requires_tau_grad : Boolean indicating whether the threshold should be optimized
        lr : learning rate for the optimization of the threshold
        """
        super(ExpectedProfit, self).__init__()
        self.k = k
        self.requires_tau_grad = requires_tau_grad
        self.tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32), requires_grad=requires_tau_grad)
        self.optimal_tau = self.tau.item()
        self.lr = lr

    def forward(self, y_pred, y_true):
        values = torch.tensor([0], dtype=torch.float32).to(y_pred.device)
        positions = torch.heaviside(y_pred - self.optimal_tau, values) - torch.heaviside(-y_pred - self.optimal_tau, values)

        profit = positions * y_true

        return profit.mean() / max_profit(y_true)

    def smooth_forward(self, y_pred, y_true):
        sig_pos = torch.sigmoid(self.k * (y_pred - self.tau))
        sig_neg = torch.sigmoid(self.k * (-y_pred - self.tau))
        positions = sig_pos - sig_neg

        profit = positions * y_true

        return profit.mean() / max_profit(y_true)

    def optimize_tau(self, y_pred, y_true, steps=100):
        optimizer = torch.optim.Adam([self.tau], lr=self.lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -self.smooth_forward(y_pred, y_true)
            loss.backward()
            optimizer.step()

        self.optimal_tau = self.tau.item()

def max_profit(y_true):
    return y_true.abs().mean()