from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.model import Model


class LassoRegression(Model):
    """Lasso Regression model class."""
    def __init__(self):
        """Initialize the Lasso Regression model."""
        model = Lasso()
        super().__init__(model=model, typ="regression")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Lasso Regression model to the data and update parameters."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Lasso Regression model."""
        return self.model.predict(X)
