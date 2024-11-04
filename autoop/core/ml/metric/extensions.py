import numpy as np
from metric import Metric


class MeanSquaredError(Metric):
    r"""
    Calculates the mean squared error.

    Mean Squared Error=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2$.
    """

    @property
    def _evaluate(self, predictions, ground_truth) -> float:
        n_predictions = len(predictions)
        count = float(0)
        for i in range(0, n_predictions):
            error = predictions[i] - ground_truth[i]
            count += np.square(error)
        return count / n_predictions


class Accuracy(Metric):
    r"""
    Calculates the accuracy.

    $\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}[\hat{y}^{(i)}=y^{i}]$
    """

    @property
    def _evaluate(self, predictions, ground_truth) -> float:
        n_predictions = len(predictions)
        count = 0
        for i in range(0, n_predictions):
            if predictions[i] == ground_truth[i]:
                count += 1
        return count / n_predictions


class AccuracyInterval(Metric):
    r"""
    Calculates the accuracy with an acceptance interval of size 2*x.

    $\text{Accuracy} =
    \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}[\hat{y}^{(i)} in [y^{i}-x, y^{i}+x]]$
    """

    @property
    def _evaluate(self, predictions, ground_truth, interval: float = 0) -> float:
        n_predictions = len(predictions)
        count = 0
        for i in range(0, n_predictions):
            if predictions[i] in range(
                ground_truth[i] - interval, ground_truth[i] + interval
            ):
                count += 1

        return count / n_predictions
