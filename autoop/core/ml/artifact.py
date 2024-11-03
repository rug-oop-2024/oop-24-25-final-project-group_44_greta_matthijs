from typing import Any, Dict, Tuple

from pydantic import BaseModel


class Artifact(BaseModel):
    """Base class for all artifacts."""

    typ: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}
    data: bytes = b""

    @property
    def typ(self) -> str:
        """Getter for _type."""
        return self.typ

    @typ.setter
    def typ(self, typ: str):
        """Setter for _type."""
        self.typ = typ

    @property
    def args(self) -> Tuple[Any, ...]:
        """Getter for _args."""
        return self.args

    @args.setter
    def args(self, args: Tuple[Any, ...]):
        """Setter for _args."""
        self.args = args

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Getter for _kwargs."""
        return self.kwargs

    @kwargs.setter
    def kwargs(self, kwargs: Dict[str, Any]):
        """Setter for _kwargs."""
        self.kwargs = kwargs

    @property
    def data(self) -> bytes:
        """Getter for _data, returns decoded bytes."""
        return self.data

    @data.setter
    def data(self, data: bytes):
        """Setter for _data, encodes data in base64."""
        self.data = data

    def read(self) -> bytes:
        """Read the artifact."""
        return self.data

    def save(self, data: bytes) -> bytes:
        """Save the artifact."""
        self.data = data
        return self.data
