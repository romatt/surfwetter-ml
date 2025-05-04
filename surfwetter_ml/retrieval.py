import datetime as dt
import logging
from pathlib import Path
from typing import Literal

import click
import isodate
import numpy as np
import xarray as xr
from earthkit.data import config
from meteodatalab import ogd_api
from meteodatalab.operators import regrid
from rasterio.crs import CRS

logging.basicConfig(format=" %(name)s :: %(levelname)-8s :: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

REST_API = "https://data.geo.admin.ch/api/stac/v1/"
NWP_DIR = "/mnt/nwp2"
PARAMETERS = ["VMAX_10M", "U_10M", "V_10M", "T_2M", "TD_2M", "HZEROCL", "DURSUN", "PMSL", "TOT_PREC"]
MODELS_API = {"ICON1": "ch.meteoschweiz.ogd-forecasting-icon-ch1", "ICON2": "ch.meteoschweiz.ogd-forecasting-icon-ch2"}
MODELS_ML = {
    "ICON1": {"name": "ogd-forecasting-icon-ch1", "start": 0, "stop": 34, "freq": 3, "distance": 0.01},
    "ICON2": {"name": "ogd-forecasting-icon-ch2", "start": 34, "stop": 121, "freq": 6, "distance": 0.02},
}

config.set("cache-policy", "temporary")


def get_api_request(model: Literal["ICON1", "ICON2"], init_time: dt.datetime, parameter: str, lead_time: int) -> list:
    # Convert arguments to expected type for API
    get_ref_time = init_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    get_lead_time = isodate.duration_isoformat(dt.timedelta(hours=lead_time))

    control = ogd_api.Request(
        collection=MODELS_ML[model]["name"],
        variable=parameter,
        reference_datetime=get_ref_time,
        perturbed=False,
        horizon=get_lead_time,
    )
    ensemble = ogd_api.Request(
        collection=MODELS_ML[model]["name"],
        variable=parameter,
        reference_datetime=get_ref_time,
        perturbed=True,
        horizon=get_lead_time,
    )
    return [control, ensemble]


def load_forecast(model: Literal["ICON1", "ICON2"], param: str, init_time: dt.datetime) -> xr.DataArray:
    da_list = []
    for lead_time in range(MODELS_ML[model]["start"], MODELS_ML[model]["stop"]):
        logger.info("Loading %s init %s param %s for lead-time %s", model, init_time.strftime("%Y%m%d%H"), param, lead_time)
        api_request = get_api_request(model, init_time, param, lead_time)
        combined_da = build_forecast_step(api_request)  # Download data and combine ctrl with ensemble
        da_list.append(regrid_forecast(combined_da, model))  # Re-project and reduce size

    # Combine times, remove unused attributes, and convert to dataset
    full_forecast = xr.concat(da_list, dim="lead_time")
    del full_forecast.attrs["metadata"]
    del full_forecast.attrs["parameter"]
    del full_forecast.attrs["geography"]
    return full_forecast


def write_forecast(data: xr.Dataset, model: Literal["ICON1", "ICON2"], param: str, init_time: dt.datetime) -> None:
    # Store file on disk
    store_dir = Path(NWP_DIR, init_time.strftime("%Y%m%d_%H"))
    Path(store_dir).mkdir(parents=True, exist_ok=True)

    # test writing to zarr
    # file_name = f"{store_dir}/{model}-{init_time.strftime("%Y%m%d%H")}-{param}.zarr"
    # ds.to_zarr(file_name)

    file_name = f"{store_dir}/{model}-{init_time.strftime('%Y%m%d%H')}-{param}.nc"
    logger.info("Writing %s to disk", file_name)

    # Define datatypes
    encoding_dict = {}
    coord_names = list(data.coords)
    if "ref_time" in coord_names:
        encoding_dict["ref_time"] = {"dtype": np.double}
    if "valid_time" in coord_names:
        encoding_dict["valid_time"] = {"dtype": np.double}
    if "lead_time" in coord_names:
        encoding_dict["lead_time"] = {"dtype": np.double}
    if "eps" in coord_names:
        encoding_dict["eps"] = {"dtype": np.double}

    # Use float32 and compress param
    for variable in list(data.data_vars):
        encoding_dict[variable] = {"dtype": np.float32, "zlib": True}

    data.to_netcdf(path=file_name, engine="h5netcdf", encoding=encoding_dict)


def da_to_ds(data: xr.DataArray, param: str) -> xr.Dataset:
    logger.info("Converting dataarray to dataset for %s", param)
    # Strip unnecessary dimensions and re-order
    data = data.squeeze()
    data = data.transpose("lead_time", "eps", "y", "x")

    # Generate nice dataset
    ds = xr.Dataset(
        {
            param: (("valid_time", "eps", "lat", "lon"), data.data),
        },
        coords={
            "valid_time": data.valid_time.data,
            "eps": data.eps.data,
            "lat": data.lat[:, 0].data,
            "lon": data.lon[0].data,
            "ref_time": data.ref_time.data,
        },
    )

    # Add attributes for ensemble members
    ds["eps"].attrs = data["eps"].attrs

    return ds


def build_forecast_step(requests: list) -> xr.DataArray:
    # Load forecast data
    da_list = []
    for request in requests:
        da = ogd_api.get_from_ogd(request)
        da_list.append(da)

    # Combine control & ensemble
    return xr.concat(da_list, dim="eps")


def regrid_forecast(data: xr.DataArray, model: Literal["ICON1", "ICON2"]) -> xr.DataArray:
    # Define the target grid extent and resolution
    # xmin, xmax = -0.817, 18.183  # Longitude bounds
    # ymin, ymax = 41.183, 51.183  # Latitude bounds
    # nx, ny = 950, 500  # Number of grid points in x and y

    xmin, xmax = 5.303, 11.003  # Longitude bounds
    ymin, ymax = 45.303, 48.003  # Latitude bounds

    distance = MODELS_ML[model]["distance"]
    nx, ny = round((xmax - xmin) / distance), round((ymax - ymin) / distance)  # Number of grid points in x and y

    # Create a regular lat/lon grid using EPSG:4326
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

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
    hour_remainder = utc_now.hour % MODELS_ML[model]["freq"]
    if hour_remainder == 0:
        test_init = utc_now
    else:
        test_init = utc_now - dt.timedelta(hours=hour_remainder)

    # Build API request for the last timestep of the model init to test
    test_request = get_api_request(model, test_init, PARAMETERS[0], MODELS_ML[model]["stop"] - 1)
    try:
        _ = ogd_api.get_from_ogd(test_request[0])
        return test_init
    except ValueError:
        # When forecast is not available, return an init "freq" hours earlier
        logger.info("Forecast at %s not available yet!", test_init.strftime("%Y%m%d%H"))
        return test_init - dt.timedelta(hours=MODELS_ML[model]["freq"])


@click.command()
@click.option("--model", "-m", default="ICON1")
@click.option("--init", "-i", default=dt.datetime.now(tz=dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H"))
def process_forecast(model: Literal["ICON1", "ICON2"], init: str):
    if init == "latest":
        init_time = get_latest_init(model)
    else:
        init_time = dt.datetime.strptime(init, "%Y%m%d%H")

    for param in PARAMETERS:
        # Skip files which are already available
        store_dir = Path(NWP_DIR, init_time.strftime("%Y%m%d_%H"))
        full_path = Path(store_dir, f"{model}-{init_time.strftime('%Y%m%d%H')}-{param}.nc")
        if Path.is_file(full_path):
            logger.warning("File %s already dowloaded, skipping!", full_path)
            continue
        da = load_forecast(model, param, init_time)
        ds = da_to_ds(da, param)
        write_forecast(ds, model, param, init_time)


if __name__ == "__main__":
    process_forecast()
