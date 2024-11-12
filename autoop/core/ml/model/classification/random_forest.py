import numpy as np

from sklearn.ensemble import RandomForestClassifier

from autoop.core.ml.model.model import Model


class RandomForestModel(Model):
    """Support Vector Classifier model class."""

    def __init__(self):
        """Initialize the Support Vector Classifier model."""
        model = RandomForestClassifier()
        super().__init__(model=model, typ="classification")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Support Vector Classifier model to the data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Support Vector Classifier model."""
        return self.model.predict(X)
