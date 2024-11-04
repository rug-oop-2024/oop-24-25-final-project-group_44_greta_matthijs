from sklearn.linear_model import LogisticRegression
import numpy as np
from autoop.core.ml.model.model import Model

class LogisticRegressionModel(Model):
    """Logistic Regression model class."""
    def __init__(self):
        """Initialize a Logistic Regression model."""
        model = LogisticRegression()
        super().__init__(model=model)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Logistic Regression model to the data and update parameters."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Logistic Regression model."""
        return self.model.predict(X)