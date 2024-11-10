from abc import ABC, abstractmethod

import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_percentage_error",
    "precision",
    "log_loss",
    "recall",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    """
    Get a metric by name.

    Args:
        name (str): Name of the metric. One of the METRICS.

    Currently only two metrics are supported:
        - mean_squared_error
        - accuracy
    """
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "log_loss":
        return LogLoss()
    elif name == "mean_absolute_percentage_error":
        return MeanAbsolutePercentageError()
    else:
        raise ValueError(
            f"\nUnknown metric: {name} \n Supported metrics are: {list(METRICS)}"
        )


def _positives_counter(positive, prediction):
    _check_positive(positive)

    if positive == "pos_int" and prediction > 0:
        return True

    if positive == "neg_int" and prediction < 0:
        return True

    if positive is True and prediction is True:
        return True

    if positive is False and prediction is False:
        return True

    if positive is None and (prediction is True or prediction < 0):
        return True

    return False


def _check_positive(positive):
    options = ["pos_int", "neg_int", True, False, None]
    if positive in options:
        return True
    raise TypeError("Invalid argument for 'positive'.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate the metric."""
        pass

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the metric."""
        return self(predictions, ground_truth)


class MeanSquaredError(Metric):
    r"""
    Calculates the mean squared error.

    Mean Squared Error=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2$.
    """

    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Measure the mean squared error of the model."""
        return np.mean((ground_truth - predictions) ** 2)


class Accuracy(Metric):
    r"""
    Calculates the accuracy.

    $\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}[\hat{y}^{(i)}=y^{i}]$
    """

    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Measure the accuracy model."""
        n_predictions = len(predictions)
        count = 0
        for i in range(n_predictions):
            if predictions[i] == ground_truth[i]:
                count += 1
        return count / n_predictions


class MeanAbsolutePercentageError(Metric):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    MAPE = (1/n) * sum(|(ground_truth - predictions) / ground_truth|) * 100
    """

    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Measures the MAPE of the model."""
        predictions = np.asarray(predictions).flatten()
        ground_truth = np.asarray(ground_truth).flatten()

        if predictions.shape[0] != ground_truth.shape[0]:
            raise ValueError("Predictions and ground_truth must have the same length.")

        non_zero_mask = ground_truth != 0
        if not np.any(non_zero_mask):
            raise ValueError(
                "Mean Absolute Percentage Error is undefined "
                "for ground_truth values of zero."
            )

        mape = (
            np.mean(
                np.abs(
                    (ground_truth[non_zero_mask] - predictions[non_zero_mask])
                    / ground_truth[non_zero_mask]
                )
            )
            * 100
        )
        return mape


class Precision(Metric):
    """
    Calculates the precision.

    {positve} should be in ["pos_int", "neg_int", True, False]
        default is set for True values, positve int

    p = True positive / all positives
    """

    def __call__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        positive: str | None | bool = None,
    ) -> float:
        """Measure the precision of the model."""
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

    def __call__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        positive: str | None | bool = True,
    ) -> float:
        """Measure the recall of the model."""
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

    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Measure the log loss of the model."""
        clip_size = 1e-15
        predictions = np.clip(predictions, clip_size, 1 - clip_size)
        log_loss = np.sum(
            ground_truth * np.log(predictions)
            + (1 - ground_truth) * np.log(1 - predictions)
        )
        return -log_loss / len(ground_truth)
