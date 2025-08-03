"""This module provides simple functionality to extract direct model output parameters at defined sites and
derive a defined number of forecast targets.

Currently predicting temperature, precipitation, wind speed, wind direction, wind gusts and duration of sunshine.

For each parameter, a defined number of quantiles is extracted from the full model ensemble, e.g. 0.5Q = Median

Wind data is pre-processed and accumulated field de-aggregated.
"""

import datetime as dt
import io
import json
import logging
import os
import re
from ftplib import FTP
from pathlib import Path

import click
import numpy as np
import xarray as xr

from surfwetter_ml import CONFIG
from surfwetter_ml.config.setting import SiteSettings
from surfwetter_ml.config.setting import TargetSettings
from surfwetter_ml.plot import plot_ICON1
from surfwetter_ml.util import write_forecast

logger = logging.getLogger(__name__)


@click.command()
@click.option("--init_icon1", "-i1")
@click.option("--init_icon2", "-i2")
def predict(init_icon1: str | None = None, init_icon2: str | None = None):
    if init_icon1 is None:
        # Load the latest fully available ICON1 & ICON2 forecast
        init_icon1, init_icon2 = lookup_latest_forecast()

    # Perform pre-processing steps
    pre_process_forecast(init_icon1, init_icon2)

    # Iterate over forecasting sizes
    for site in CONFIG.forecast.sites:
        for target in CONFIG.forecast.targets:
            # Define file name and check if forecast has already been processed
            file_name = f"{site.name}-{init_icon1}-{target.parameter}.json"
            if Path.is_file(Path(CONFIG.data, init_icon1, file_name)):
                logging.warning("Prediction %s for %s already exists", target.parameter, site.name)
                continue

            logging.info("Extracting %s predictions for %s", target.parameter, site.name)
            icon1_fcst = xr.open_dataset(Path(CONFIG.data, init_icon1, f"ICON1-{init_icon1}-{target.parameter}.nc"))
            icon2_fcst = xr.open_dataset(Path(CONFIG.data, init_icon2, f"ICON2-{init_icon2}-{target.parameter}.nc"))

            # Deaggregate forecast if needed
            if target.accumulated:
                icon1_fcst = icon1_fcst.diff(dim="valid_time", label="lower")
                icon2_fcst = icon2_fcst.diff(dim="valid_time", label="lower")

            # Compute statistics
            icon1_quant = compute_quantiles(icon1_fcst, site, target)
            icon2_quant = compute_quantiles(icon2_fcst, site, target)

            # Combine short and mid-term forecast
            forecast = combine_forecasts(icon1_quant, icon2_quant)

            # Add metadata to forecast
            forecast = add_metadata(forecast, target)

            # Upload forecast to FTP server
            upload_forecast(forecast, file_name)

    # Generate plots
    logging.info("Plot forecast")
    locations = [cfg.location for cfg in CONFIG.plot]
    for location in locations:
        file_name = f"lake_{location}.webp"
        if Path.is_file(Path(CONFIG.data, init_icon1, file_name)):
            logging.warning("Plot %s already exists", file_name)
            continue
        plot_ICON1(init_icon1, location)
        upload_plot(init_icon1, file_name)

def pre_process_forecast(init_icon1: str, init_icon2: str) -> None:
    """Run pre-processing steps for foreacst, only needed once per forecast

    Parameters
    ----------
    init_icon1 : str
        Latest ICON-CH1-EPS initialization
    init_icon2 : str
        Latest ICON-CH2-EPS initialization
    """
    pre_process_wind("ICON1", init_icon1)
    pre_process_wind("ICON2", init_icon2)


def pre_process_wind(model: str, init: str) -> None:
    if Path.is_file(Path(CONFIG.data, init, f"{model}-{init}-WIND_DIR.nc")) and Path.is_file(
        Path(CONFIG.data, init, f"{model}-{init}-WIND_SPEED.nc")
    ):
        logging.warning("Wind for %s init %s is already pre-processed", model, init)
        return

    logging.info("Pre-processing wind for %s init %s", model, init)
    u = xr.open_dataset(Path(CONFIG.data, init, f"{model}-{init}-U_10M.nc"))
    v = xr.open_dataset(Path(CONFIG.data, init, f"{model}-{init}-V_10M.nc"))

    # Compute and store wind direction
    wind_dir = np.degrees(np.arctan2(u.U_10M.values, v.V_10M.values)) % 360
    icon_dir = xr.zeros_like(u)
    icon_dir["WIND_DIR"] = (icon_dir.dims, wind_dir)
    icon_dir = icon_dir.drop_vars("U_10M")
    write_forecast(icon_dir, model, "WIND_DIR", dt.datetime.strptime(init, CONFIG.dtfmt))

    # Compute and store wind speed
    wind_speed = np.sqrt((u.U_10M.values)**2, (v.V_10M.values)**2)
    icon_speed = xr.zeros_like(u)
    icon_speed["WIND_SPEED"] = (icon_speed.dims, wind_speed)
    icon_speed = icon_speed.drop_vars("U_10M")
    write_forecast(icon_speed, model, "WIND_SPEED", dt.datetime.strptime(init, CONFIG.dtfmt))


def combine_forecasts(icon1: xr.DataArray, icon2: xr.DataArray) -> xr.DataArray:
    icon1_valid = icon1.valid_time.values
    icon2_valid = icon2.valid_time.values
    combined_valid_time = np.intersect1d(icon1_valid, icon2_valid)
    icon2_select = np.setxor1d(combined_valid_time, icon2_valid)

    # Compute weighted averages for the last 3 hours of overlap (25%, 50%, 75%)
    overlap = []
    weights = [0.25, 0.5, 0.75]
    for i, overlap_time in enumerate(combined_valid_time[-3:]):
        overlap.append(icon1.sel(valid_time=overlap_time) * (1 - weights[i]) + icon2.sel(valid_time=overlap_time) * weights[i])

    overlap_da = xr.concat(overlap, dim="valid_time")
    return xr.concat([icon1.sel(valid_time=icon1_valid[:-3]), overlap_da, icon2.sel(valid_time=icon2_select)], dim="valid_time")


def add_metadata(forecast: xr.DataArray, target: TargetSettings) -> xr.DataArray:
    """Add meta data to xarray

    Parameters
    ----------
    forecast : xr.DataArray
        The forecast
    target : TargetSettings
        The target settings

    Returns
    -------
    xr.DataArray
        Updated xarray
    """
    forecast.attrs["unit"] = target.unit
    forecast.attrs["description"] = target.description
    return forecast


def upload_forecast(forecast: xr.DataArray, file_name: str) -> None:
    """Convert forecast from xarray to JSON and upload bytes to FTP server. Additionally, store JSON file locally

    Parameters
    ----------
    forecast : xr.DataArray
        Dataarray holding the forecast
    file_name : str
        Name of JSON file to generate
    """

    logger.info("Uploading %s to FTP server...", file_name)

    # Define time format
    forecast["valid_time"] = forecast.valid_time.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Convert forecast to JSON
    forecast_dict = forecast.to_dict()
    forecast_json = json.dumps(forecast_dict)

    # Convert forecast to xml bytes
    forecast_bytes = io.BytesIO()
    forecast_bytes.write(forecast_json.encode())
    forecast_bytes.seek(0)

    # Store files also locally
    with Path.open(Path(CONFIG.data, file_name.split("-")[1], file_name), "w") as f:
        f.write(forecast_json)

    date_str = re.search("[0-9]{10}-", file_name)
    remote_fn = file_name.replace(date_str[0], "")

    # Connect to server and upload forecast
    ftp_server = FTP(CONFIG.ftp.host, CONFIG.ftp.user, CONFIG.ftp.password)
    ftp_server.storbinary(f"STOR {remote_fn}", forecast_bytes)

    # Close the Connection
    ftp_server.quit()


def upload_plot(model_init: str, fname: str = "lake_lucerne.webp") -> None:
    # Upload image
    ftp_server = FTP(CONFIG.ftp.host, CONFIG.ftp.user, CONFIG.ftp.password)
    with Path.open(Path(CONFIG.data, model_init, fname), "rb") as file:
        ftp_server.storbinary(f"STOR {fname}", file)
    ftp_server.quit()


def lookup_latest_forecast() -> tuple[str, str]:
    """Find the latest fully downloaded forecasts for ICON1 & ICON2

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

    logging.info("Latest forecasts ICON1: %s ICON2: %s", init_icon1, init_icon2)

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
    icon1_files = [file for file in files if file.startswith(model)]
    icon1_params = [file.split("-")[-1].replace(".nc", "") for file in icon1_files]  # Get list of parameters, remove trailing .nc
    if set(icon1_params) >= set(CONFIG.nwp.parameters):
        return True
    else:
        return False


def compute_quantiles(data: xr.Dataset, site: SiteSettings, target: TargetSettings) -> xr.DataArray:
    """Extract parameter at required station and compute quantiles from forecast ensemble

    Parameters
    ----------
    data : xr.Dataset
        Model forecast ensemble
    site : SiteSettings
        Site settings with `lon` and `lat` attributes
    target : TargetSettings
        Target settings with information about quantiles to compute

    Returns
    -------
    xr.DataArray
        Requested quantiles of parameter extracted at required station
    """
    local_forecast = data[target.parameter].sel(lon=site.lon, lat=site.lat, method="nearest")  # Select location and parameter
    statistics = []
    for quantile in target.quantiles:
        local_statistic = local_forecast.quantile(q=quantile, dim="eps")
        statistics.append(local_statistic)

    return xr.concat(statistics, dim="quantile")


if __name__ == "__main__":
    predict()
