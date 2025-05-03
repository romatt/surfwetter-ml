import os

import yaml
from pydantic import BaseModel

from typing import Any

class SubscriptableBaseModel(BaseModel):

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

class ParamSettings(SubscriptableBaseModel):
    pass


class NWPSettings(SubscriptableBaseModel):

    parameters: list[ParamSettings]


class ServiceSettings(BaseModel):
    nwp: NWPSettings

    @staticmethod
    def load(setting_files: list) -> "ServiceSettings":
        """Load from settings file"""
        config_data: dict = {}
        for path in setting_files:
            with open(os.path.join(os.path.dirname(__file__), path), "r", encoding='UTF-8') as yaml_file:
                config_data.update(yaml.safe_load(yaml_file))
        # """Load from environment"""
        for setting_name in config_data:
            for setting in config_data[setting_name]:
                env_var = os.environ.get(f'{setting_name}__{setting}')
                if env_var:
                    config_data[setting_name][setting] = env_var
        return ServiceSettings(**config_data)
