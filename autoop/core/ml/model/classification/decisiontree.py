from sklearn.tree import DecisionTreeClassifier
import numpy as np
from autoop.core.ml.model.model import Model

class DecisionTreeModel(Model):
    """Decision Tree model class."""
    def __init__(self, parameters: dict = None):
        """Initialize a Decision Tree model."""
        model = DecisionTreeClassifier()
        super().__init__(model=model)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Decision Tree model to the data and update parameters."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Decision Tree model."""
        return self.model.predict(X)