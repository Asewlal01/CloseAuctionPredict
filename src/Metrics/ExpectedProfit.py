import torch.nn as nn
import torch

class ExpectedProfit(nn.Module):
    def __init__(self, k=50, init_tau=0.5, requires_tau_grad=True, lr=1e-3):
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
        return profit.mean()

    def smooth_forward(self, y_pred, y_true):
        sig_pos = torch.sigmoid(self.k * (y_pred - self.tau))
        sig_neg = torch.sigmoid(self.k * (-y_pred - self.tau))
        positions = sig_pos - sig_neg

        profit = positions * y_true

        return profit.mean()

    def optimize_tau(self, y_pred, y_true, steps=100):
        optimizer = torch.optim.Adam([self.tau], lr=self.lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -self.smooth_forward(y_pred, y_true)
            loss.backward()
            optimizer.step()

        self.optimal_tau = self.tau.item()