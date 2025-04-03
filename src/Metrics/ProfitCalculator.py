import torch
import torch.nn as nn

class ProfitCalculator(nn.Module):
    """
    This class can be used to compute profit statistics for a given set of predictions and true values.
    """
    def __init__(self, short_threshold=0, long_threshold=0, find_thresholds=False, lr=0.001):
        """
        Initializes the ProfitCalculator.

        Parameters
        ----------
        short_threshold : Threshold to enter a short position.
        long_threshold : Threshold to enter a long position.
        find_thresholds : Whether to find the optimal thresholds.
        lr : Learning rate for the optimizer.
        """
        super(ProfitCalculator, self).__init__()
        self.find_thresholds = find_thresholds

        # Save the thresholds as tensors
        short_threshold = torch.tensor([short_threshold])
        long_threshold = torch.tensor([long_threshold])

        if find_thresholds:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.short_threshold = nn.Parameter(short_threshold, requires_grad=find_thresholds)
        self.long_threshold = nn.Parameter(long_threshold, requires_grad=find_thresholds)

    def forward(self, y_pred, y_true):
        """
        This function computes the mean profit for a given set of predictions and true values.

        Parameters
        ----------
        y_pred : Predicted values.
        y_true : True values.

        Returns
        -------
        Mean profit made by the strategy
        """

        profits = compute_profit(y_pred, y_true, self.short_threshold.item(), self.long_threshold.item())
        return profits.mean() / maximum_profit(y_true)

    def approximated_profit(self, y_pred, y_true):
        """
        This function computes the mean profit for a given set of predictions and true values. It approximates the
        profit function by using a sigmoid instead of a Heaviside function. This allows for differentiability which is
        needed to find the optimal thresholds.

        Parameters
        ----------
        y_pred : Predicted values.
        y_true : True values.

        Returns
        -------
        Approximated mean profit made by the strategy.
        """
        profits = approximated_profit(y_pred, y_true, self.short_threshold, self.long_threshold)
        return profits.mean()

    def compounded_profit(self, y_pred, y_true):
        """
        This function computes the compounded profit for a given set of predictions and true values.
        It assumes that it keeps reinvesting the profits made by the strategy, hence compounding the returns.

        Parameters
        ----------
        y_pred : Predicted values.
        y_true : True values.

        Returns
        -------
        Compounded profit made by the strategy.
        """
        profits = compute_profit(y_pred, y_true, self.short_threshold.item(), self.long_threshold.item())
        return (1 + profits).prod()

    def find_thresholds(self, y_pred, y_true, steps=100):
        """
        This function finds the optimal thresholds for the short and long positions.

        Parameters
        ----------
        y_pred : Predicted values.
        y_true : True values.
        steps : Number of steps to optimize the thresholds.

        Returns
        -------
        Optimal thresholds and the best profit.
        """
        if not self.find_thresholds:
            raise ValueError("This model was not initialized to find thresholds.")
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = -self.approximated_profit(y_pred, y_true)
            loss.backward()
            self.optimizer.step()


def compute_profit(y_pred, y_true, short_threshold, long_threshold):
    """
    This function computes the profit for a given set of predictions and true values.

    Parameters
    ----------
    y_pred : Predicted values.
    y_true : True values.
    short_threshold : Threshold to enter a short position.
    long_threshold : Threshold to enter a long position.

    Returns
    -------
    The profit made by the strategy.
    """
    # Heaviside function needs values when the input is zero, hence we give it a value of zero.
    values = torch.tensor([0], dtype=y_pred.dtype, device=y_pred.device)

    # A short position corresponds to -1 and a long position corresponds to 1.
    short_positions = torch.heaviside(short_threshold - y_pred, values)
    long_positions = torch.heaviside(y_pred - long_threshold, values)
    positions = long_positions - short_positions

    return positions * y_true

def maximum_profit(y_true):
    """
    This function computes the maximum profit possible mean profit for a given set of true values.

    Parameters
    ----------
    y_true : True values.

    Returns
    -------
    The maximum profit that can be made.
    """
    return y_true.abs().mean()

def approximated_profit(y_pred, y_true, short_threshold, long_threshold):
    """
    This function computes the profit for a given set of predictions and true values. It approximates the profit
    function by using a sigmoid instead of a Heaviside function. This allows for differentiability.

    Parameters
    ----------
    y_pred : Predicted values.
    y_true : True values.
    short_threshold : Threshold to enter a short position.
    long_threshold : Threshold to enter a long position.

    Returns
    -------
    The profit made by the strategy.
    """

    # A short position corresponds to -1 and a long position corresponds to 1.
    short_positions = torch.sigmoid(y_pred - short_threshold)
    long_positions = torch.sigmoid(y_pred - long_threshold)
    positions = long_positions - short_positions

    return positions * y_true