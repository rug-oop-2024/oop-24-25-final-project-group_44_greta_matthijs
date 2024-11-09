from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoop.core.ml.model.model import Model


class KNearestNeighborsModel(Model):
    """K-Nearest Neighbors model class."""
    def __init__(self, parameters: dict = None):
        """Initialize a K-Nearest Neighbors model."""
        model = KNeighborsClassifier()
        super().__init__(model=model, typ="classification")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the K-Nearest Neighbors model to the data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the K-Nearest Neighbors model."""
        return self.model.predict(X)
