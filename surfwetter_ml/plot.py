import datetime as dt
import logging
import os
import subprocess

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from surfwetter_ml import CONFIG

logger = logging.getLogger(__name__)


def plot_ICON1(forecast: str, location: str) -> None:
    logger.info("Plotting lake %s for %s", location, forecast)
    title, extent, mesh_thres, line_thres = get_plot_settings(location)

    gusts = xr.open_dataset(f"{CONFIG.data}/{forecast}/ICON1-{forecast}-VMAX_10M.nc")
    gusts["VMAX_10M"] = gusts["VMAX_10M"] * 1.944  # convert windspeed from m/s to knots

    # Load rain
    rain = xr.open_dataset(f"{CONFIG.data}/{forecast}/ICON1-{forecast}-TOT_PREC.nc")
    rain_t0 = rain.isel(valid_time=0)
    rain_diff = rain.diff(dim="valid_time", label="upper")
    rain = xr.concat([rain_t0, rain_diff], dim='valid_time')

    # Convert to local time
    gusts = set_timezone(gusts, "valid_time")
    rain = set_timezone(rain, "valid_time")

    # Compute probability to exceed 14 and 24 knots
    exceed_14 = xr.where(gusts.VMAX_10M >= mesh_thres, 1, 0)
    prob_14 = exceed_14.sum(dim="eps") / 11
    exceed_20 = xr.where(gusts.VMAX_10M >= line_thres, 1, 0)
    prob_20 = exceed_20.sum(dim="eps") / 11

    # Reduce size of data to speed up plotting
    prob_14 = prob_14.sel(lat=slice(extent[2] - 0.1, extent[3] + 0.1), lon=slice(extent[0] - 0.1, extent[1] + 0.1))
    prob_20 = prob_20.sel(lat=slice(extent[2] - 0.1, extent[3] + 0.1), lon=slice(extent[0] - 0.1, extent[1] + 0.1))
    rain_median = rain.TOT_PREC.median(dim="eps").sel(
        lat=slice(extent[2] - 0.1, extent[3] + 0.1), lon=slice(extent[0] - 0.1, extent[1] + 0.1)
    )

    # Define offset for lead-time, number of rows and coloums and figure height depending on model init
    # For early runs (and very late runs) start plots at 9AM
    if gusts.valid_time.data.hour[0] <= 8 or gusts.valid_time.data.hour[0] >= 17:
        lt_offset = np.where(exceed_14.valid_time.data.hour == 10)[0][0]
        ncols, nrows = 3, 3
        figheight = 18
    elif gusts.valid_time.data.hour[0] <= 11:  # otherwise at 12AM
        lt_offset = np.where(exceed_14.valid_time.data.hour == 11)[0][0]
        ncols, nrows = 3, 3
        figheight = 18
    elif gusts.valid_time.data.hour[0] <= 14:  # or at 3PM
        lt_offset = np.where(exceed_14.valid_time.data.hour == 14)[0][0]
        ncols, nrows = 2, 3
        figheight = 14

    # Initialize figure
    plt.rcParams.update({"font.size": 15})
    fig = plt.figure(figsize=(30, figheight))  # width, height
    naxs = ncols * nrows
    axs = naxs * [GeoAxes]

    # Iterate over lead-times
    for idx in range(naxs):
        # Select data and convert to probabilities
        plot_gusts_14 = prob_14.isel(valid_time=(idx + lt_offset)) * 100
        plot_gusts_20 = prob_20.isel(valid_time=(idx + lt_offset)) * 100
        plot_time = gusts.valid_time.data[idx + lt_offset]
        plot_rain = rain_median.sel(valid_time=plot_time)

        # Initialize plot
        axs[idx] = fig.add_subplot(3, 3, idx + 1, projection=ccrs.Mercator())  # slow AF
        # axs[idx] = fig.add_subplot(3, 3, idx + 1, projection=ccrs.PlateCarree())

        # Plot 14 knots gusts probabilities
        im = plot_gusts_14.plot(
            ax=axs[idx], cmap="CMRmap_r", levels=np.arange(10, 100, 10), transform=ccrs.PlateCarree(), add_colorbar=False
        )

        # Overlay 20 knots gust probabilities
        plot_gusts_20.plot.contour(
            ax=axs[idx], levels=np.array([25, 50, 75]), transform=ccrs.PlateCarree(), cmap="winter", add_colorbar=False
        )

        # Add contours for rain
        plot_rain.plot.contourf(
            ax=axs[idx],
            levels=np.array([0, 1]),
            hatches=["", ".."],
            transform=ccrs.PlateCarree(),
            colors=["none", "#3a40e880"],
            add_colorbar=False,
        )
        # rain_contours.set_edgecolor("red")

        axs[idx].title.set_text(f"{plot_time.strftime('%d.%m.%Y %H:%M')}LT T+{idx + lt_offset}h")
        add_overlay(axs[idx], extent)

    # Adjust padding
    fig.subplots_adjust(top=0.95, bottom=0.08, wspace=0.02, hspace=0.02)

    # Add labels, colorbar, and annotate plot
    cbar_ax = fig.add_axes([0.36, 0.05, 0.3, 0.03])  # left, bottom, weight, height
    fig.colorbar(mappable=im, cax=cbar_ax, location="bottom", label=f"Böen > {mesh_thres} Knoten (%)")
    fig.legend(
        handles=[Patch(color="#3a40e880", hatch=".", label=">1mm Regen")], loc="upper left", bbox_to_anchor=(0.2, 0.98), frameon=False
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0000ff", lw=1, label="25%"),
            Line2D([0], [0], color="#007fbf", lw=1, label="50%"),
            Line2D([0], [0], color="#00ff80", lw=1, label="75%"),
        ],
        ncols=3,
        loc="upper left",
        bbox_to_anchor=(0.7, 1.0),
        frameon=False,
        title=f"Wahrscheinlichkeit Böen > {line_thres} Knoten",
    )

    fig.text(0.9, 0.05, "Datenquelle: MeteoSchweiz", ha="right")
    fig.text(0.13, 0.05, "surfwetter.ch", ha="left", weight="bold", size="30", alpha=0.5)
    fig.text(0.9, 0.07, f"ICON-CH1-EPS Modellauf {gusts.valid_time[0].dt.strftime('%d.%m.%Y %H:%M').data} UTC", ha="right")
    fig.suptitle(title, weight="bold", size=25)
    fig.savefig(f"{CONFIG.data}/{forecast}/lake_{location}.png", dpi=150, bbox_inches="tight")

    # Make systemcall to convert to webp...
    os.chdir(f"{CONFIG.data}/{forecast}")
    subprocess.run(["cwebp", "-q", "80", f"lake_{location}.png", "-o", f"lake_{location}.webp"])


def set_timezone(ds: xr.Dataset, time_var: str, timezone: str = "Europe/Zurich") -> xr.Dataset:
    time_index = ds.valid_time.to_index()
    time_utc = time_index.tz_localize(dt.UTC)
    time_lt = time_utc.tz_convert(pytz.timezone(timezone))
    ds[time_var] = time_lt
    return ds


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
    if lakes_within_bound.empty:
        ax.add_feature(cfeature.LAKES,
                       edgecolor='dodgerblue',
                       facecolor='#ffffff00')
    else:
        ax.add_geometries(lakes_within_bound.geometry, crs=ccrs.PlateCarree(), facecolor="#ffffff00", edgecolor="dodgerblue")

    # Add points for surf spots
    for site in CONFIG.forecast.sites:
        ax.scatter(site.lon, site.lat, transform=ccrs.PlateCarree(), marker="*", c=["magenta"], label=site.name)

    # Overlay grid
    gl = ax.gridlines(draw_labels=True, color="#4c4c4c", linestyle=(0, (5, 5)))
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False
    ax.set_extent(extent)


def get_plot_settings(location: str) -> tuple[str, list, int, int]:
    """Load plotting settings

    Parameters
    ----------
    location : str
        Location to plot

    Returns
    -------
    tuple[str, list]
        Plot title and extent as well as mesh and line thresholds
    """

    for plot_config in CONFIG.plot:
        if plot_config.location == location:
            return plot_config.title, plot_config.extent, plot_config.mesh_thres, plot_config.line_thres


@click.command()
@click.option("--forecast", "-f")
def main_wrapper(forecast: str):
    plot_ICON1(forecast)


if __name__ == "__main__":
    main_wrapper()
