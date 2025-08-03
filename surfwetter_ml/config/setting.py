from typing import Any

from pydantic import BaseModel


class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class RegridSettings(SubscriptableBaseModel):
    xmin: float
    xmax: float
    """Longitude bounds for re-gridding"""

    ymin: float
    ymax: float
    """Latitude bounds for re-gridding"""


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
    regrid: RegridSettings


class TargetSettings(SubscriptableBaseModel):
    parameter: str
    """Parameter to predict"""

    description: str
    """Description of parameter to predict"""

    quantiles: list[float]
    """Which statistics should be derived from the model ensemble"""

    accumulated: bool
    """Whether the NWP data is accumulated"""

    unit: str
    """Unit of foreacst"""


class SiteSettings(SubscriptableBaseModel):
    name: str
    """Name of prediction site"""

    desc: str
    """Long name or description of site"""

    lon: float
    """Longitude of prediction site"""

    lat: float
    """Latitude of prediction site"""


class ForecastSettings(SubscriptableBaseModel):
    sites: list[SiteSettings]
    targets: list[TargetSettings]


class FTPSettings(SubscriptableBaseModel):
    host: str
    """FTP host"""

    user: str
    """FTP username"""

    password: str
    """FTP password"""

class PlotSettings(SubscriptableBaseModel):
    location: str
    """Location name"""

    title: str
    """Title of plot"""

    extent: list[float, float, float, float]
    """Plotting bounds"""

    mesh_thres: int
    """Mesh threshold for wind gust probabilities"""

    line_thres: int
    """Line threshold for wind gust probabilities"""


class LibrarySettings(BaseModel):
    nwp: NWPSettings
    """Settings for NWP data"""

    data: str
    """Storage location of NWP data"""

    dtfmt: str
    """Date format"""

    api: str
    """STAC API"""

    forecast: ForecastSettings
    """Settings for predicitons"""

    ftp: FTPSettings
    """Setting for connection to FTP server"""

    plot: list[PlotSettings]
    """Settings for plotting"""
