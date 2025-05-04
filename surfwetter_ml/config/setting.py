from typing import Any

from pydantic import BaseModel


class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ModelSettings(SubscriptableBaseModel):
    name: str
    """The model name as expected by the STAC API"""

    start: int
    """First hour of extract"""

    stop: int
    """Last hour of extraction"""

    freq: int
    """Model update frequency"""

    distance: float
    """Model resolution after re-gridding in WGS84"""


class ModelList(SubscriptableBaseModel):
    ICON1: ModelSettings
    ICON2: ModelSettings


class NWPSettings(SubscriptableBaseModel):
    parameters: list[str]
    models: ModelList


class LibrarySettings(BaseModel):
    nwp: NWPSettings
    """Settings for forecasts"""

    data: str
    """Storage location of NWP data"""

    api: str
    """STAC API"""
