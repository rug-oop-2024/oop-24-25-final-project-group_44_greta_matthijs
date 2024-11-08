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
    raise NotImplementedError("To be implemented.")

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

    def __call__(self, predictions, ground_truth) -> float:
        return self.evaluate(predictions, ground_truth)

    @property
    @abstractmethod
    def _evaluate(self, predictions, ground_truth) -> float:
        pass
