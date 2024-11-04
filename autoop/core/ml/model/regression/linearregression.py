from sklearn.linear_model import LinearRegression
import numpy as np
from autoop.core.ml.model.model import Model

class LRModel(Model):
    """Linear Regression model class."""
    def __init__(self):
        """Initialize the Linear Regression model."""
        model = LinearRegression()
        super().__init__(model=model)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Linear Regression model to the data and update parameters."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Linear Regression model."""
        return self.model.predict(X)