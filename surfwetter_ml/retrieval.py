import datetime as dt
import logging
from pathlib import Path
from typing import Literal

import click
import isodate
import xarray as xr
from earthkit.data import config
from meteodatalab import ogd_api
from meteodatalab.operators import regrid
from rasterio.crs import CRS

from surfwetter_ml import CONFIG
from surfwetter_ml.util import da_to_ds
from surfwetter_ml.util import write_forecast

logger = logging.getLogger(__name__)

# Set Cache for downloaded data
config.set("cache-policy", "temporary")


def get_api_request(model: Literal["ICON1", "ICON2"], init_time: dt.datetime, parameter: str, lead_time: int) -> list:
    # Convert arguments to expected type for API
    get_ref_time = init_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    get_lead_time = isodate.duration_isoformat(dt.timedelta(hours=lead_time))

    control = ogd_api.Request(
        collection=CONFIG.nwp.models[model].name,
        variable=parameter,
        reference_datetime=get_ref_time,
        perturbed=False,
        horizon=get_lead_time,
    )
    ensemble = ogd_api.Request(
        collection=CONFIG.nwp.models[model].name,
        variable=parameter,
        reference_datetime=get_ref_time,
        perturbed=True,
        horizon=get_lead_time,
    )
    return [control, ensemble]


def load_forecast(model: Literal["ICON1", "ICON2"], param: str, init_time: dt.datetime) -> xr.DataArray:
    da_list = []
    for lead_time in range(CONFIG.nwp.models[model].start, CONFIG.nwp.models[model].stop):
        logger.info("Loading %s init %s param %s for lead-time %s", model, init_time.strftime(CONFIG.dtfmt), param, lead_time)
        api_request = get_api_request(model, init_time, param, lead_time)
        combined_da = build_forecast_step(api_request)  # Download data and combine ctrl with ensemble
        da_list.append(regrid_forecast(combined_da, model))  # Re-project and reduce size

    # Combine times, remove unused attributes, and convert to dataset
    full_forecast = xr.concat(da_list, dim="lead_time")
    del full_forecast.attrs["metadata"]
    del full_forecast.attrs["parameter"]
    del full_forecast.attrs["geography"]
    return full_forecast


def build_forecast_step(requests: list) -> xr.DataArray:
    # Load forecast data
    da_list = []
    for request in requests:
        da = ogd_api.get_from_ogd(request)
        da_list.append(da)

    # Combine control & ensemble
    return xr.concat(da_list, dim="eps")


def regrid_forecast(data: xr.DataArray, model: Literal["ICON1", "ICON2"]) -> xr.DataArray:
    """Re-grid forecast to WGS84

    Parameters
    ----------
    data : xr.DataArray
        Input data
    model : Literal['ICON1', 'ICON2']
        Model type, either 'ICON1' or 'ICON2'

    Returns
    -------
    xr.DataArray
        Re-gridded data
    """
    # Define the target grid extent and resolution
    xmin, xmax = CONFIG.nwp.regrid.xmin, CONFIG.nwp.regrid.xmax
    ymin, ymax = CONFIG.nwp.regrid.ymin, CONFIG.nwp.regrid.ymax
    distance = CONFIG.nwp.models[model].distance
    nx, ny = round((xmax - xmin) / distance), round((ymax - ymin) / distance)  # Number of grid points in x and y

    # Create a regular lat/lon grid using EPSG:4326
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx+1, ny+1, xmin, xmax, ymin, ymax)

    # Remap ICON native grid data to the regular grid
    return regrid.iconremap(data, destination)


def get_latest_init(model: Literal["ICON1", "ICON2"]) -> dt.datetime:
    """Function to get the latest available model initialization by attempting to download a CTRL run

    Parameters
    ----------
    model : Literal['ICON1', 'ICON2']
        Model type, either 'ICON1' or 'ICON2'

    Returns
    -------
    dt.datetime
        Initialization time
    """

    # Derive the latest possible forecast based on the model initialization frequency
    utc_now = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    hour_remainder = utc_now.hour % CONFIG.nwp.models[model].freq
    if hour_remainder == 0:
        test_init = utc_now
    else:
        test_init = utc_now - dt.timedelta(hours=hour_remainder)

    # Build API request for the last timestep of the model init to test
    test_request = get_api_request(model, test_init, CONFIG.nwp.parameters[0], CONFIG.nwp.models[model].stop - 1)
    try:
        _ = ogd_api.get_from_ogd(test_request[0])
        return test_init
    except ValueError:
        # When forecast is not available, return an init "freq" hours earlier
        logger.info("Forecast at %s not available yet!", test_init.strftime(CONFIG.dtfmt))
        return test_init - dt.timedelta(hours=CONFIG.nwp.models[model].freq)


@click.command()
@click.option("--model", "-m", default="ICON1")
@click.option("--init", "-i", default=dt.datetime.now(tz=dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0).strftime(CONFIG.dtfmt))
def process_forecast(model: Literal["ICON1", "ICON2"], init: str):
    if init == "latest":
        init_time = get_latest_init(model)
    else:
        init_time = dt.datetime.strptime(init, CONFIG.dtfmt)

    for param in CONFIG.nwp.parameters:
        # Skip files which are already available
        store_dir = Path(CONFIG.data, init_time.strftime(CONFIG.dtfmt))
        full_path = Path(store_dir, f"{model}-{init_time.strftime(CONFIG.dtfmt)}-{param}.nc")
        if Path.is_file(full_path):
            logger.warning("File %s already dowloaded, skipping!", full_path)
            continue
        da = load_forecast(model, param, init_time)
        ds = da_to_ds(da, param)
        write_forecast(ds, model, param, init_time)


if __name__ == "__main__":
    process_forecast()
