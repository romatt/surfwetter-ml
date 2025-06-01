import datetime as dt
import logging
import os
import subprocess

import cartopy.crs as ccrs
import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

from surfwetter_ml import CONFIG

logger = logging.getLogger(__name__)


def plot_ICON1(forecast: str) -> None:
    logger.info("Plotting lake_lucerne.png for %s", forecast)
    file = xr.open_dataset(f"{CONFIG.data}/{forecast}/ICON1-{forecast}-VMAX_10M.nc")
    file["VMAX_10M"] = file["VMAX_10M"] * 1.944  # convert windspeed from m/s to knots

    # Convert to local time
    time_index = file.valid_time.to_index()
    time_utc = time_index.tz_localize(dt.UTC)
    time_lt = time_utc.tz_convert(pytz.timezone("Europe/Zurich"))
    file["valid_time"] = time_lt

    # Compute probability to exceed 14 and 24 knots
    exceed_14 = xr.where(file.VMAX_10M >= 14, 1, 0)
    prob_14 = exceed_14.sum(dim="eps") / 11
    exceed_24 = xr.where(file.VMAX_10M >= 24, 1, 0)
    prob_24 = exceed_24.sum(dim="eps") / 11

    # Reduce size needed for plotting
    prob_14 = prob_14.sel(lat=slice(46.7, 47.2), lon=slice(8, 9))
    prob_24 = prob_24.sel(lat=slice(46.7, 47.2), lon=slice(8, 9))

    # Define offset for lead-time, number of rows and coloums and figure height depending on model init
    # For early runs (and very late runs) start plots at 9AM
    if file.valid_time.data.hour[0] <= 8 or file.valid_time.data.hour[0] >= 17:
        lt_offset = np.where(exceed_14.valid_time.data.hour == 10)[0][0]
        ncols, nrows = 3, 3
        figheight = 18
    elif file.valid_time.data.hour[0] <= 11:  # otherwise at 12AM
        lt_offset = np.where(exceed_14.valid_time.data.hour == 11)[0][0]
        ncols, nrows = 3, 3
        figheight = 18
    elif file.valid_time.data.hour[0] <= 14:  # or at 3PM
        lt_offset = np.where(exceed_14.valid_time.data.hour == 14)[0][0]
        ncols, nrows = 2, 3
        figheight = 14

    # Initialize figure
    plt.rcParams.update({"font.size": 15})
    fig = plt.figure(figsize=(30, figheight))  # width, height
    naxs = ncols * nrows
    axs = naxs * [GeoAxes]
    extent = [8.1, 8.8, 46.8, 47.1]

    # Iterate over lead-times
    for idx in range(naxs):
        # Select data and convert to probabilities
        plot_data = prob_14.isel(valid_time=(idx + lt_offset)) * 100
        plot_time = file.valid_time.data[idx + lt_offset]

        # Initialize plot
        axs[idx] = fig.add_subplot(3, 3, idx + 1, projection=ccrs.Mercator())  # slow AF
        # axs[idx] = fig.add_subplot(3, 3, idx + 1, projection=ccrs.PlateCarree())

        im = plot_data.plot(ax=axs[idx], cmap="CMRmap_r", levels=np.arange(10, 100, 10), transform=ccrs.PlateCarree(), add_colorbar=False)

        axs[idx].title.set_text(f"{plot_time.strftime('%d.%m.%Y %H:%M')}LT T+{idx + lt_offset}h")
        add_overlay(axs[idx], extent)

    # Adjust padding
    fig.subplots_adjust(top=0.95, bottom=0.08, wspace=0.02, hspace=0.02)

    # Add common colorbar
    cbar_ax = fig.add_axes([0.333, 0.05, 0.3, 0.03])
    fig.colorbar(mappable=im, cax=cbar_ax, location="bottom", shrink=0.4, label="Böen > 14 Knoten (%)")
    fig.suptitle(f"ICON-CH1-EPS Modellauf {file.valid_time[0].dt.strftime('%d.%m.%Y %H:%M').data} UTC", fontsize=20)
    fig.savefig(f"{CONFIG.data}/{forecast}/lake_lucerne.png", dpi=150, bbox_inches="tight")

    # Make systemcall to convert to webp...
    os.chdir(f"{CONFIG.data}/{forecast}")
    subprocess.run(["cwebp", "-q", "80", "lake_lucerne.png", "-o", "lake_lucerne.webp"])


def add_overlay(ax, extent: list):
    """
    Overlay features, coordinate grid and set extent of plot

    Parameters
    ----------
    ax :
        Axes
    extent : str or list
        Set the plots extent with coordinates west, east, south, north]
    """

    ch_lakes = gpd.read_file("/home/roman/projects/surfwetter-ml/src/lakes")
    # Filter within view to speed up rendering
    lakes_within_bound = ch_lakes.cx[extent[0] : extent[1], extent[2] : extent[3]]
    ax.add_geometries(lakes_within_bound.geometry, crs=ccrs.PlateCarree(), facecolor="#ffffff00", edgecolor="dodgerblue")

    # Add points for surf spots
    ax.scatter(8.617259408225175, 46.91610571992836, transform=ccrs.PlateCarree(), marker="*", c=["magenta"])
    ax.scatter(8.312032, 46.967248, transform=ccrs.PlateCarree(), marker="*", c=["magenta"])
    ax.scatter(8.600355, 46.919067, transform=ccrs.PlateCarree(), marker="*", c=["magenta"])

    # Overlay grid
    gl = ax.gridlines(draw_labels=True, color="#4c4c4c", linestyle=(0, (5, 5)))
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False
    ax.set_extent(extent)


@click.command()
@click.option("--forecast", "-f")
def main_wrapper(forecast: str):
    plot_ICON1(forecast)


if __name__ == "__main__":
    main_wrapper()
