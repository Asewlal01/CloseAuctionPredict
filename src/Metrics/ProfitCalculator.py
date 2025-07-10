import torch

class ProfitCalculator:
    """
    This class can be used to compute profit statistics for a given set of predictions and true values.
    """
    def __init__(self, daily_predictions: list[torch.Tensor], threshold: float=0.5, weighted: bool=False):
        """
        Initializes the ProfitCalculator with daily predictions, a threshold, and a flag for weighted calculations.

        Parameters
        ----------
        daily_predictions : Predictions of the model for each day. Should be a list of torch.Tensor with the first
        column being the predicted values and the second column being the true values.
        threshold : Threshold of entering a position.
        weighted : If True, the size of the position is weighted by the predicted value.
        """
        self.returns = []
        for prediction in daily_predictions:
            if isinstance(prediction, torch.Tensor) and prediction.dim() == 2 and prediction.size(1) == 2:
                daily_return = compute_daily_return(prediction, threshold, weighted)
                self.returns.append(daily_return)

            else:
                raise ValueError("Each prediction must be a 2D tensor with two columns: predicted and true values.")

        # Allows for easier computations
        self.returns = torch.tensor(self.returns)

    def win_rate(self) -> float:
        """
        Computes the win rate of the predictions. Defined as the percentage of days where the return was positive.

        Returns
        -------
        The win rate as a float.
        """
        positive_returns = (self.returns > 1).sum().float().item()
        total_returns = len(self.returns)

        return positive_returns / total_returns

    def expected_return(self) -> float:
        """
        Computes the expected return of the predictions. Defined as the average log-return across all days.

        Returns
        -------
        The expected return as a float.
        """
        log_returns = torch.log(self.returns)

        return log_returns.mean().item()

    def sharpe_ratio(self) -> float:
        """
        Computes the Sharpe ratio of the predictions. Based on log-returns, similar to the expected return.

        Returns
        -------
        The Sharpe ratio as a float.
        """
        log_returns = torch.log(self.returns)
        mean_return = log_returns.mean().item()
        std_return = log_returns.std().item()

        return mean_return / std_return

    def maximum_drawdown(self) -> float:
        """
        Computes the maximum drawdown of the predictions. Based on log-returns, similar to the expected return.

        Returns
        -------
        The maximum drawdown as a float.
        """
        log_returns = torch.log(self.returns)

        # Calculate cumulative returns
        cumulative_returns = torch.cumsum(log_returns, dim=0)

        # Find the running maximum
        running_max = torch.cummax(cumulative_returns, dim=0)[0]

        # Calculate drawdowns
        drawdowns = running_max - cumulative_returns

        # Return the maximum drawdown
        return drawdowns.max().item()

    def total_cumulative_return(self) -> float:
        """
        Computes the total cumulative return of the predictions. Defined as the product of daily returns minus one.

        Returns
        -------
        The total cumulative return as a float.
        """
        cumulative_returns = torch.prod(self.returns)

        return cumulative_returns.item() - 1

    def cumulative_return(self) -> torch.Tensor:
        """
        Computes the cumulative return of the predictions. Defined as the product of daily returns minus one.

        Returns
        -------
        The cumulative return as a float.
        """
        cumulative_returns = torch.cumprod(self.returns, dim=0)

        return cumulative_returns - 1


def compute_daily_return(predictions: torch.Tensor, threshold: float, weighted: bool) -> float:
        """
        Computes the daily return based on predictions and a threshold.

        Parameters
        ----------
        predictions : A tensor containing the predicted values as the first column and the true values as the second column.
        threshold : The threshold for entering a position.
        weighted : If True, the size of the position is weighted by the predicted value.

        Returns
        -------
        The daily return as a tensor
        """
        # Determine the positions we take based on the predictions and threshold
        positions = determine_positions(predictions, threshold)
        if (positions == 0).all():
            return 1    # No investments so zero growth (growth = 1)

        # Computing the return based on the positions taken
        returns = weighted_returns(predictions, positions, weighted)

        return returns.sum().item()


def determine_positions(predictions: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Determines the positions to take based on predictions and a threshold.

        Parameters
        ----------
        predictions : A tensor containing the predicted values as the first column and the true values as the second column.
        threshold : The threshold for entering a position.

        Returns
        -------
        A tensor indicating the positions to take (1 for long, -1 for short, 0 for no position).
        """
        y_pred, y_true = predictions[:, 0], predictions[:, 1]
        y_prob = torch.sigmoid(y_pred)  # Assuming y_pred is logits

        positions = torch.zeros_like(y_prob)
        positions[y_prob > threshold] = 1  # Long position
        positions[y_prob < -threshold] = -1  # Short position

        return positions

def stock_return(predictions: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Computes the stock return based on predictions and positions.

        Parameters
        ----------
        predictions : A tensor containing the predicted values as the first column and the true values as the second column.
        positions : A tensor indicating the positions to take (1 for long, -1 for short, 0 for no position).

        Returns
        -------
        The stock return as a tensor.
        """
        y_true = predictions[:, 1]
        returns = y_true * positions

        return returns

def weighted_returns(predictions: torch.tensor, positions: torch.tensor, weighted=False) -> torch.Tensor:
    """
    Computes the weighted returns based on predictions and positions.

    Parameters
    ----------
    predictions : A tensor containing the predicted values as the first column and the true values as the second column.
    positions : A tensor indicating the positions to take (1 for long, -1 for short, 0 for no position).
    weighted : If True, the size of the position is weighted by the predicted value.

    Returns
    -------
    The weighted stock return as a tensor.
    """
    # Not weighted so equal division
    weights = torch.ones_like(positions) / len(positions)
    if weighted:
        entered_positions = positions != 0
        y_pred = predictions[:, 0]
        y_pred_abs = torch.abs(y_pred[entered_positions])
        pos_weights = torch.softmax(y_pred_abs, dim=0)
        weights = torch.zeros_like(y_pred)
        weights[entered_positions] = pos_weights

    returns = stock_return(predictions, positions)
    returns *= weights

    return returns


