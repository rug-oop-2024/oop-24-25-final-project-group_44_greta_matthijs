from abc import abstractmethod, ABC
import numpy as np
from copy import deepcopy


class Model(ABC):
    """Model class to represent a machine learning model."""
    def __init__(self, model: object, typ: str):
        """Initialize the model."""
        self._parameters = {}
        self._model = model
        self._type = typ

    @property
    def parameters(self) -> dict:
        """Getter for _parameters."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """Setter for _parameters."""
        self._parameters = parameters

    @property
    def model(self) -> object:
        """Getter for _model."""
        return self._model

    @model.setter
    def model(self, model: object) -> None:
        """Setter for _model."""
        self._model = model

    @property
    def type(self) -> str:
        """Getter for _type."""
        return self._type

    @type.setter
    def type(self, typ: str) -> None:
        """Setter for _type."""
        self._type = typ

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the data."""
        pass
