from enum import Enum
from pydantic import BaseModel


class LabelEnum(str, Enum):
    safe = "safe"
    unsafe = "unsafe"


class OutPrompt(BaseModel):
    query: str
    label: LabelEnum
