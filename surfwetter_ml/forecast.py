import io
import os
import re
from ftplib import FTP
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from dict2xml import dict2xml

from surfwetter_ml import CONFIG
from surfwetter_ml.config.setting import SiteSettings
from surfwetter_ml.config.setting import TargetSettings


def predict():
    # Iterate over forecasting sizes
    for site in CONFIG.forecast.sites:
        for target in CONFIG.forecast.targets:
            # Load the latest fully available ICON1 & ICON2 forecast
            init_icon1, init_icon2 = get_latest_forecast()
            icon1_fcst = xr.open_dataset(Path(CONFIG.data, init_icon1, f"ICON1-{init_icon1}-{target.parameter}.nc"))
            icon2_fcst = xr.open_dataset(Path(CONFIG.data, init_icon2, f"ICON2-{init_icon2}-{target.parameter}.nc"))

            # Compute statistics
            icon1_quant = compute_quantiles(icon1_fcst, site, target)
            icon2_quant = compute_quantiles(icon2_fcst, site, target)

            # Combine short and long-term forecast
            forecast = combine_forecasts(icon1_quant, icon2_quant)

            # Upload forecast to FTP server
            upload_forecast(forecast, site, target)


def combine_forecasts(icon1: xr.DataArray, icon2: xr.DataArray) -> xr.DataArray:

    icon1_valid = icon1.valid_time.values
    icon2_valid = icon2.valid_time.values
    combined_valid_time = np.intersect1d(icon1_valid, icon2_valid)
    icon2_select = np.setxor1d(combined_valid_time, icon2_valid)

    # Compute weighted averages for the last 3 hours of overlap (25%, 50%, 75%)
    overlap = []
    weights = [0.25, 0.5, 0.75]
    for i, overlap_time in enumerate(combined_valid_time[-3:]):
        overlap.append(icon1.sel(valid_time=overlap_time)* (1-weights[i]) + icon2.sel(valid_time=overlap_time)* weights[i])

    overlap_da = xr.concat(overlap, dim="valid_time")
    return xr.concat([icon1.sel(valid_time=icon1_valid[:-3]), overlap_da, icon2.sel(valid_time=icon2_select)], dim="valid_time")


def upload_forecast(forecast: xr.DataArray, site: SiteSettings, target: TargetSettings) -> None:

    # Convert forecast to XML
    forecast_dict = forecast.to_dict()
    forecast_xml = dict2xml(forecast_dict)

    # Store forecast
    forecast_bytes = io.BytesIO()
    forecast_bytes.write(forecast_xml.encode())
    forecast_bytes.seek(0)

    # Define file name
    filename = f"{site.name}-{pd.to_datetime(forecast.valid_time[0].data).strftime(CONFIG.dtfmt)}-{target.parameter}.xml"

    # Connect to server
    ftp_server = FTP(CONFIG.ftp.host, CONFIG.ftp.user, CONFIG.ftp.password)

    ftp_server.storbinary(f'STOR {filename}', forecast_bytes)

    # Close the Connection
    ftp_server.quit()


def get_latest_forecast() -> tuple[str, str]:
    """Get the latest fully available forecasts for ICON1 & ICON2

    Returns
    -------
    tuple[str, str]
        Folder with forecasts
    """

    # List all folders in data directory
    folders = os.listdir(CONFIG.data)

    # Only keep folders that follow the pattern YYYYMMDDHH
    pattern = re.compile("[0-9]{10}")
    folders = [folder for folder in folders if pattern.match(folder)]
    folders.sort(reverse=True)

    # Check if all requried files are available
    init_icon1 = ""
    init_icon2 = ""
    for folder in folders:
        files = os.listdir(Path(CONFIG.data, folder))
        if init_icon2 == "" and match_files("ICON2", files):
            init_icon2 = folder
        if init_icon1 == "" and match_files("ICON1", files):
            init_icon1 = folder

    return init_icon1, init_icon2


def match_files(model: str, files: list) -> bool:
    """Check if required forecast parameters are available in directory

    Parameters
    ----------
    model : str
        Model name, either 'ICON1' or 'ICON2'
    files : list
        List of files in directory

    Returns
    -------
    bool
        Whether all files are available or not
    """
    required_params = [target.parameter for target in CONFIG.forecast.targets]
    icon1_files = [file for file in files if file.startswith(model)]
    icon1_params = [file.split("-")[-1].replace(".nc", "") for file in icon1_files]  # Get list of parameters, remove trailing .nc
    if set(icon1_params) >= set(required_params):
        return True
    else:
        return False


def compute_quantiles(data: xr.Dataset, site: SiteSettings, target: TargetSettings) -> xr.DataArray:
    local_forecast = data[target.parameter].sel(lon=site.lon, lat=site.lat, method="nearest")  # Select location and parameter
    statistics = []
    for quantile in target.quantiles:
        local_statistic = local_forecast.quantile(q=quantile, dim="eps")
        statistics.append(local_statistic)

    return xr.concat(statistics, dim="quantile")


def combine_forecasts(icon1: xr.DataArray, icon2: xr.DataArray) -> xr.DataArray:
    pass


def make_xml():
    pass


def upload_xml():
    pass


if __name__ == "__main__":
    predict()
