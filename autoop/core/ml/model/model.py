from abc import abstractmethod, ABC
import numpy as np
from copy import deepcopy

class Model(ABC):
    """Model class to represent a machine learning model."""
    def __init__(self, model: object):
        """Initialize the model."""
        self.parameters = {}
        self.model = model

    @property
    def parameters(self) -> dict:
        """Getter for _parameters."""
        return deepcopy(self.parameters)
    
    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """Setter for _parameters."""
        self.parameters = parameters
    
    @property
    def model(self) -> object:
        """Getter for _model."""
        return self.model
    
    @model.setter
    def model(self, model: object) -> None:
        """Setter for _model."""
        self.model = model
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the data."""
        pass
    
