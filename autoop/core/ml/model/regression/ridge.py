from sklearn.linear_model import Ridge
import numpy as np
from autoop.core.ml.model.model import Model

class RidgeModel(Model):
    """Ridge Regression model class."""

    def __init__(self):
        """Initialize the Ridge Regression model."""
        model = Ridge()
        super().__init__(model=model)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Ridge Regression model to the data and update parameters."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Ridge Regression model."""
        return self.model.predict(X)