import uuid
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class Artifact(BaseModel):
    """Base class for all artifacts."""

    type: str
    name: str
    asset_path: str
    version: str = "1.0.0"
    data: bytes = b""
    tags: List[str] = []
    metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "file_size": len(data)
        }
    id: str = str(uuid.uuid4())

    @property
    def typ(self) -> str:
        """Getter for _type."""
        return self.typ

    @typ.setter
    def typ(self, typ: str):
        """Setter for _type."""
        self.typ = typ

    @property
    def data(self) -> bytes:
        """Getter for _data, returns decoded bytes."""
        return self._data

    @data.setter
    def data(self, data: bytes):
        """Setter for _data, encodes data in base64."""
        self._data = data

    def read(self) -> bytes:
        """Read the artifact."""
        return self.data

    def save(self, data: bytes) -> bytes:
        """Save the artifact."""
        self.data = data
        return self.data
