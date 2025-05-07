import datetime as dt
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from surfwetter_ml import CONFIG

logger = logging.getLogger(__name__)


def write_forecast(data: xr.Dataset, model: Literal["ICON1", "ICON2"], param: str, init_time: dt.datetime) -> None:
    # Store file on disk
    store_dir = Path(CONFIG.data, init_time.strftime(CONFIG.dtfmt))
    Path(store_dir).mkdir(parents=True, exist_ok=True)

    # test writing to zarr
    # file_name = f"{store_dir}/{model}-{init_time.strftime(CONFIG.dtfmt)}-{param}.zarr"
    # ds.to_zarr(file_name)

    file_name = f"{store_dir}/{model}-{init_time.strftime(CONFIG.dtfmt)}-{param}.nc"
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
