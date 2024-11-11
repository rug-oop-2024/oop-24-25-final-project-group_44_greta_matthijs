from typing import List

import numpy as np

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.

    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    features = []
    for name in data.columns:
        if data[name].dtype == "object":
            features.append(Feature(name=name, type="categorical"))
        else:
            features.append(Feature(name=name, type="numerical"))
    return features


def improved_detect_features(dataset: Dataset) -> List[Feature]:
    """Improved version of detect_feature_types function.

    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    features = []
    for name in data.columns:
        col_data = data[name]
        unique_values = np.unique(col_data.dropna())

        if col_data.dtype == "object" or len(unique_values) <= 10:
            features.append(Feature(name=name, type="categorical"))
        else:
            features.append(Feature(name=name, type="numerical"))

    return features
