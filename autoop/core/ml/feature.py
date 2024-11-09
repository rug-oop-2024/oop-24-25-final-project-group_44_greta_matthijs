from typing import Literal

from pydantic import BaseModel


class Feature(BaseModel):
    """Feature class to represent a feature in a dataset."""

    name: str
    typ: Literal["categorical", "numerical"]

    @property
    def name(self) -> str:
        """Getter for _name."""
        return self.name

    @name.setter
    def name(self, name: str):
        """Setter for _name."""
        self.name = name

    @property
    def typ(self) -> Literal["categorical", "numerical"]:
        """Getter for _type."""
        return self.typ

    @typ.setter
    def typ(self, typ: Literal["categorical", "numerical"]):
        """Setter for _type."""
        self.typ = typ

    def __str__(self):
        """Return the string representation of the feature."""
        return f"Feature(name={self._name}, type={self._type})"
