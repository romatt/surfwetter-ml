import datetime as dt
import json
import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
import shapely.wkt
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


def process_shapes():
    import geopandas as gpd

    # Load raw data
    lines = gpd.read_file("src/TLM_GEWAESSER/swissTLM3D_TLM_STEHENDES_GEWAESSER.shp")

    # Convert linestrings to polygons
    polygons = lines.polygonize()

    # Keep only lakes with a certain area
    large_lakes = polygons[polygons.area > 10000]

    # Re-project
    large_lakes = large_lakes.to_crs(4326)

    # Store to disk
    large_lakes.to_file("src/lakes")


def lake_outlines():
    """
    Additional shapefiles:
    - https://deims.org/f30007c4-8a6e-4f11-ab87-569db54638fe
    - https://deims.org/58036d71-8141-40c3-a0f2-50b8bd1bcddc
    """

    # Load swiss lakes
    with open("/home/roman/projects/surfwetter-ml/src/Lakes_new.json") as f:
        lakes_list = json.load(f)

    # Keep only lakes that have predictions
    keep_lakes = [
        "Urnersee",
        "Alpnachersee",
        "Murtensee",
        "Silvaplanersee",
        "Greifensee",
        "Zürichsee",
        "Sihlsee",
        "Walensee",
        "Neuenburgersee",
        "Untersee",
        "Zugersee",
        "Lake Maggiore"
    ]
    lakes_forecasted = [lake for lake in lakes_list if lake["Name"] in keep_lakes]

    # Convert to geopandas DF
    df = pd.DataFrame(lakes_forecasted)
    gdf_lakes = gpd.GeoDataFrame(df["Name"], geometry=shapely.wkt.loads(df["GeometryCH1903+"]), crs="EPSG:2056")

    # Re-project to WGGS84
    gdf_lakes = gdf_lakes.to_crs(4326)

    # Add Gardasee
    garda_json = open("/home/roman/projects/surfwetter-ml/src/gardasee.geojson")
    gdf_garda = gpd.read_file(garda_json, columns="geometry")
    gdf_garda.insert(1, "Name", ["Gardasee"])
    #    gdf_garda.drop(["deimsid", "id", "field_elevation_avg_value"], axis=1, inplace=True)
    #    gdf_garda.rename(columns={"name": "Name"}, inplace=True)
    gdf_combined = pd.concat([gdf_lakes, gdf_garda])

    # Add Comersee
    como_json = open("/home/roman/projects/surfwetter-ml/src/como.geojson")
    gdf_como = gpd.read_file(como_json, columns="geometry")
    gdf_como.insert(1, "Name", ["Comersee"])
    gdf_combined = pd.concat([gdf_combined, gdf_como])

    # Reduce decimals
    gdf_combined.geometry = gdf_combined.geometry.set_precision(grid_size=0.000001)

    # Save objects
    gdf_combined.to_file("/home/roman/projects/surfwetter-ml/src/fcst_lakes.geojson", driver="GeoJSON")


def set_timezone(ds: xr.Dataset | xr.DataArray, time_var: str, timezone: str = "Europe/Zurich") -> xr.Dataset | xr.DataArray:
    time_index = ds.valid_time.to_index()
    try:
        time_utc = time_index.tz_localize(dt.UTC)
    except TypeError:
        logger.info("Object already tz-aware")
        time_utc = time_index
    time_lt = time_utc.tz_convert(pytz.timezone(timezone))
    ds[time_var] = time_lt
    return ds
