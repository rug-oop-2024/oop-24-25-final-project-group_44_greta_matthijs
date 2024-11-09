from abc import ABC, abstractmethod
from typing import Any

import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    match name:
        case "Mean Squared Error":
            return MeanSquaredError()
        case "Accuracy":
            return Accuracy()
        case "AccuracyInterval":
            return AccuracyInterval()
        case "Precision":
            return Precision()
        case "log_loss":
            return LogLoss()
    raise ValueError(f"No metric with name {name}")


def _positives_counter(positive, prediction):
    _check_positive(positive)

    if positive == "pos_int" and prediction > 0:
        return True

    elif positive == "neg_int" and prediction < 0:
        return True

    elif positive is True and prediction is True:
        return True

    elif positive is False and prediction is False:
        return True

    elif positive is None and (prediction is True or prediction < 0):
        return True

    return False


def _check_positive(positive):
    options = ["pos_int", "neg_int", True, False, None]
    if positive in options:
        return True
    raise TypeError("Invalid argument for 'positive'.")


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return self.evaluate(predictions, ground_truth)

    @property
    @abstractmethod
    def _evaluate(self, predictions, ground_truth) -> float:
        pass


class MeanSquaredError(Metric):
    r"""
    Calculates the mean squared error.

    Mean Squared Error=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2$.
    """

    @property
    def _evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
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
    def _evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        n_predictions = len(predictions)
        count = 0
        for i in range(n_predictions):
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
    def _evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray, interval: float = 0
    ) -> float:
        n_predictions = len(predictions)
        count = 0
        for i in range(0, n_predictions):
            if predictions[i] in range(
                ground_truth[i] - interval, ground_truth[i] + interval
            ):
                count += 1

        return count / n_predictions


class Precision(Metric):
    """
    Calculates the precision.

    {positve} should be in ["pos_int", "neg_int", True, False]
        default is set for True values, positve int

    p = True positive / all positives
    """

    def _evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        positive: str | None | bool = None,
    ) -> float:
        _check_positive(positive)

        # initialize variables
        true_positives = int(0)
        all_positives = int(0)
        n_predictions = len(predictions)

        # find all positives
        for i in range(0, n_predictions):
            if _positives_counter(positive, predictions[i]):
                all_positives += 1
                if predictions[i] == ground_truth[i]:
                    true_positives += 1

        # to prevent 0 division error
        if all_positives == 0:
            return 0

        return true_positives / all_positives


class Recall(Metric):
    """
    Calculates the recall.

     {positve} should be in ["pos_int", "neg_int", True, False]
        default is set for True values, positve int

    Recall = true positives / (true positives + false negatives)
    """

    def _evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        positive: str | None | bool = None,
    ) -> float:
        _check_positive(positive)

        true_positives = 0
        false_negatives = 0
        n_predictions = len(predictions)

        for i in range(0, n_predictions):
            if _positives_counter(positive, predictions[i]):
                if predictions[i] == ground_truth[i]:
                    true_positives += 1

            else:
                if predictions[i] != ground_truth[i]:
                    false_negatives += 1

        if true_positives == 0 and false_negatives == 0:
            return 0

        return true_positives / (true_positives + false_negatives)


class LogLoss(Metric):
    """
    Calculates the logarithmic loss.

    Log Loss= -N1∑i = 1N∑j = 1Myij⋅log(pij)
    """

    def _evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        clip_size = 1e-15
        predictions = np.clip(predictions, clip_size, 1 - clip_size)
        log_loss = np.sum(
            ground_truth * np.log(predictions)
            + (1 - ground_truth) * np.log(1 - predictions)
        )
        return -log_loss / len(ground_truth)
