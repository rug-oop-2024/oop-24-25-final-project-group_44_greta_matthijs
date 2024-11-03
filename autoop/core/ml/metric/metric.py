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


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, predictions, ground_truth) -> float:
        return self.evaluate(predictions, ground_truth)

    @property
    @abstractmethod
    def _evaluate(self, predictions, ground_truth) -> float:
        pass
