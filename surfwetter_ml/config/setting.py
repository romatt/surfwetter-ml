from typing import Any

from pydantic import BaseModel


class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ModelSettings(SubscriptableBaseModel):
    start: int
    stop: int
    freq: int
    distance: float


class ModelList(SubscriptableBaseModel):
    ICON1: ModelSettings
    ICON2: ModelSettings


class NWPSettings(SubscriptableBaseModel):
    parameters: list[str]
    model: ModelList


class LibrarySettings(BaseModel):
    nwp: NWPSettings
    data: str
    api: str
