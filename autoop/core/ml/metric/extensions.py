import numpy as np

from autoop.core.ml.metric.metric import Metric


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


class Precision(Metric):
    """
    Calculates the precision.

    {positve} should be in ["pos_int", "neg_int", True, False]
        default is set for True values, positve int

    p = True positive / all positives
    """

    def _evaluate(
        self, predictions, ground_truth, positive: str | None | bool = None
    ) -> float:
        def positives_counter(positive, prediction):
            if positive is None:
                if prediction is True or (
                    isinstance(prediction, (float, int)) and prediction > 0
                ):
                    return True

            options = ["pos_int", "neg_int", True, False, None]
            if positive not in options:
                raise TypeError("Invalid argument for 'positive'.")

            if positive == "pos_int":
                if prediction > 0:
                    return True
                return False

            elif positive == "neg_int":
                if prediction < 0:
                    return True
                return False

            elif positive is True:
                if prediction is True:
                    return True
                return False

            elif positive is False:
                if prediction is False:
                    return True
                return False

        true_positives = int(0)
        all_positives = int(0)
        n_predictions = len(predictions)
        for i in range(0, n_predictions):
            
            if positives_counter(positive, predictions[i]):
                all_positives += 1
                if predictions[i] == ground_truth[i]:
                    true_positives += 1

        if true_positives == 0 or all_positives == 0:
            return 0

        return true_positives / all_positives
