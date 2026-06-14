import glob
import logging
import os
import pickle
import re
import time
import warnings
from datetime import datetime, timedelta
from os import makedirs
from pathlib import Path

import astropy.coordinates
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pfsspy
import sunpy
from aiapy.calibrate import correct_degradation, update_pointing
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pfss.functions_data import (
    PrepHMIdaily,
    aia_correction,
    aia_download_from_date,
    hmi_daily_download,
)
from pfsspy.fieldline import ClosedFieldLines, OpenFieldLines
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time
from sunpy.map import Map
from sunpy.net import Fido, attrs
from sunpy.net import attrs as a
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.time import parse_time
from time_utils import time_of

warnings.filterwarnings("ignore")
logging.getLogger("sunpy").setLevel(logging.WARNING)
# /mnt/scratch/data/orlovsd2/sunpy/data

def get_closest_aia(date_time_obj, wavelength = 193):
    # Add this line near the top of the file, after the imports
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    aia_start = date_time_obj - timedelta(minutes=0.5)
    aia_end = date_time_obj + timedelta(minutes=0.5)
    aia_start_str = aia_start.strftime(DATE_FORMAT)[:-3]
    aia_end_str = aia_end.strftime(DATE_FORMAT)[:-3]
    
    aia_local_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data")
    aia_files = sorted(glob.glob(f"{aia_local_dir}/aia.lev1.193A_*.fits"))

    for aia_file in aia_files:
        match = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", aia_file)
        if match:
            filename_time_str = match.group(1)  # Extract timestamp from filename
            formatted_time_str = filename_time_str.replace("_", "-").replace("T", " ")  # Convert format
            filename_time = datetime.strptime(formatted_time_str, "%Y-%m-%d %H-%M-%S")  # Convert to datetime

            if aia_start <= filename_time <= aia_end:
                print(f"Using local AIA file: {aia_file}")
                return sunpy.map.Map(aia_file)  

    print(f"No local AIA file found within ±30 seconds for {date_time_obj}")
    return None  # If no valid file is found


    #max_retries = 5
    #retry_delay = 120
    
    #for attempt in range(max_retries):
    #    try:
    #        # Search for AIA data
    #        search_results = Fido.search(a.Time(aia_start_str, aia_end_str), a.Instrument.aia, a.Wavelength(wavelength*u.angstrom))
                    
    #        if search_results:
    #            # Download the AIA file
    #            aia_downloads = Fido.fetch(search_results)
                        
    #            if aia_downloads:  # Ensure at least one file was downloaded
    #                return sunpy.map.Map(aia_downloads[0])
    #            else:
    #                raise ValueError("AIA data download failed, empty list returned.")

    #        else:
    #            raise ValueError("No AIA data found for the requested time.")

    #    except Exception as e:
    #        print(f"Attempt {attempt+1} failed: {e}")
    #        time.sleep(retry_delay)

    #    print(f"Skipping AIA data for {date_time_obj} after {max_retries} failed attempts.")
    #    return None  # Return None instead of crashing

def _hmi_day_local_strict(day_time, root):
    """
    Strict local-only lookup for HMI daily synoptic map on the same day.
    Returns a prepped SunPy Map via PrepHMIdaily.
    Raises FileNotFoundError if none found.
    """
    day = Time(day_time).strftime("%Y%m%d")
    # your files are consistent: hmi.mrdailysynframe_720s.YYYYMMDD_HHMMSS_TAI.data.fits
    pat = f"hmi.mrdailysynframe_720s.{day}_*.fits"
    hits = sorted(glob.glob(str(Path(root) / pat)))
    if not hits:
        raise FileNotFoundError(f"No HMI daily synoptic file for {day} under {root}")
    chosen = hits[-1]  # if multiple times per day, take the latest
    print(f"[HMI local] using: {chosen}")
    return PrepHMIdaily(chosen)

def _hmi_local_today_vs_yesterday(day_time, root):
    """
    Local-only lookup for HMI daily synoptic map closest to `day_time`,
    comparing same-day vs previous-day files under `root`.
    Returns a prepped SunPy Map via PrepHMIdaily.
    Raises FileNotFoundError only if neither day exists locally.
    """
    t = Time(day_time)

    def _find_latest_for_date(date_str_ymd):
        # date_str_ymd is "YYYY-MM-DD"
        y, m, d = date_str_ymd.split("-")

        pats_rel = [
            f"hmi.mrdailysynframe_720s.{y}{m}{d}_*.fits",
            f"hmi.mrdailysynframe_720s.{y}{m}{d}*.fits",
            f"hmi.mrdailysynframe_720s.{y}.{m}.{d}*.fits",
            f"*mrdailysynframe*{y}{m}{d}*.fits",
            f"*mrdailysynframe*{y}.{m}.{d}*.fits",
        ]

        hits = []
        for pat in pats_rel:
            hits.extend(glob.glob(str(Path(root) / pat)))

        hits = sorted(set(hits))
        if not hits:
            return None

        chosen = str(Path(hits[-1]))  # latest if multiple
        print(f"[HMI local] candidate: {chosen}")
        return PrepHMIdaily(chosen)

    day_today = t.strftime("%Y-%m-%d")
    day_yest = (t - 1 * u.day).strftime("%Y-%m-%d")

    hmi_today = _find_latest_for_date(day_today)
    hmi_yest  = _find_latest_for_date(day_yest)

    Tt = time_of(hmi_today)
    Ty = time_of(hmi_yest)

    if (Tt is None) and (Ty is None):
        raise FileNotFoundError(f"No HMI daily synoptic file for {day_today} or {day_yest} under {root}")
    elif (Tt is not None) and (Ty is None):
        choice = hmi_today
    elif (Tt is None) and (Ty is not None):
        choice = hmi_yest
    else:
        choice = hmi_today if abs(Tt - t) <= abs(Ty - t) else hmi_yest

    print(f"[HMI local] chosen date: {time_of(choice)} for EIS time {t}")
    return choice


def get_pfss_from_map(eis_map, min_gauss = -20, max_gauss = 20, dimension = (1080, 540)):
    debug = False
    # use the closest local HMI daily map
    hmi_root = Path(r"C:\Users\domor\fyp\fyp_data\hmi_daily")
    eis_time = Time(eis_map.date)
    tag = Time(eis_map.date)
    m_hmi = _hmi_local_today_vs_yesterday(eis_time, hmi_root)

    if debug:
        print("EIS time:", eis_time)
        print("HMI chosen time:", Time(m_hmi.date))

    # helper functions for changing observer time and frame
    change_obstime = lambda x,y: SkyCoord( # x original Skycoord, y = new time
        x.replicate( # makes a copy of x
            observer=x.observer.replicate(obstime=y), # takes original observer and makes a copy of it with a new time, y.
            obstime=y # sets the new time for the copy of x
        ))


    change_obstime_frame = lambda x,y: x.replicate_without_data( #original frame, y = new time
        observer=x.observer.replicate(obstime=y), # Makes a copy of the frame without copying any coordinate data inside.
        obstime=y)

    # Resample the HMI data to a specific resolution
    m_hmi_resample = m_hmi.resample(dimension * u.pix) # .resample changes the number of pixels in the map, and stretching/compressing the coordinate system to match the new pixel size.

    # put the resampled HMI map into the EIS observation frame
    new_frame = change_obstime_frame(m_hmi_resample.coordinate_frame, eis_map.date)

    if debug:
        print("EIS observer:", eis_map.observer_coordinate)
        print("EIS obstime:", eis_map.date)
        print("HMI observer:", m_hmi_resample.observer_coordinate)
        print("HMI obstime:", m_hmi_resample.date)
        print("EIS coordinate frame:", eis_map.coordinate_frame)
        print("HMI coordinate frame:", m_hmi_resample.coordinate_frame)

    blc_hpc = SkyCoord(eis_map.bottom_left_coord.Tx - 0.1 * (eis_map.top_right_coord.Tx - eis_map.bottom_left_coord.Tx),
        eis_map.bottom_left_coord.Ty - 0.1 * (eis_map.top_right_coord.Ty - eis_map.bottom_left_coord.Ty), frame=eis_map.coordinate_frame)

    trc_hpc = SkyCoord(eis_map.top_right_coord.Tx + 0.1 * (eis_map.top_right_coord.Tx - eis_map.bottom_left_coord.Tx),
        eis_map.top_right_coord.Ty + 0.1 * (eis_map.top_right_coord.Ty - eis_map.bottom_left_coord.Ty), frame=eis_map.coordinate_frame)

    rs = eis_map.rsun_obs.to_value(u.arcsec)
    r_blc = np.sqrt(blc_hpc.Tx.to_value(u.arcsec)**2 + blc_hpc.Ty.to_value(u.arcsec)**2)
    r_trc = np.sqrt(trc_hpc.Tx.to_value(u.arcsec)**2 + trc_hpc.Ty.to_value(u.arcsec)**2)

    if debug:
        print("rsun_obs arcsec:", rs)
        print("corner r arcsec:", r_blc, r_trc)

    if r_blc > rs:
        s = 0.999 * rs / r_blc
        blc_hpc = SkyCoord(blc_hpc.Tx * s, blc_hpc.Ty * s, frame=blc_hpc.frame)

    if r_trc > rs:
        s = 0.999 * rs / r_trc
        trc_hpc = SkyCoord(trc_hpc.Tx * s, trc_hpc.Ty * s, frame=trc_hpc.frame)

    if debug:
        print("EIS corners Tx/Ty:", blc_hpc.Tx, blc_hpc.Ty, "|", trc_hpc.Tx, trc_hpc.Ty)
        print("EIS corners finite:", np.isfinite(blc_hpc.Tx.to_value(u.arcsec)), np.isfinite(blc_hpc.Ty.to_value(u.arcsec)),
            np.isfinite(trc_hpc.Tx.to_value(u.arcsec)), np.isfinite(trc_hpc.Ty.to_value(u.arcsec)))
        r_blc = np.sqrt(blc_hpc.Tx.to_value(u.arcsec)**2 + blc_hpc.Ty.to_value(u.arcsec)**2)
        r_trc = np.sqrt(trc_hpc.Tx.to_value(u.arcsec)**2 + trc_hpc.Ty.to_value(u.arcsec)**2)
        print("rsun_obs arcsec:", eis_map.rsun_obs.to_value(u.arcsec))
        print("corner r arcsec:", r_blc, r_trc)

    blc_syn = blc_hpc.transform_to(new_frame)
    trc_syn = trc_hpc.transform_to(new_frame)

    if debug:
        print("after transform_to(new_frame):", blc_syn.lon, blc_syn.lat, "|", trc_syn.lon, trc_syn.lat)
        print("transform finite:", np.isfinite(blc_syn.lon.to_value(u.deg)), np.isfinite(blc_syn.lat.to_value(u.deg)),
            np.isfinite(trc_syn.lon.to_value(u.deg)), np.isfinite(trc_syn.lat.to_value(u.deg)))

    blc_ar_synop = change_obstime(blc_syn, m_hmi_resample.date)
    trc_ar_synop = change_obstime(trc_syn, m_hmi_resample.date)

    if debug:
        print("after change_obstime:", blc_ar_synop.lon, blc_ar_synop.lat, "|", trc_ar_synop.lon, trc_ar_synop.lat)
        print("change_obstime finite:", np.isfinite(blc_ar_synop.lon.to_value(u.deg)), np.isfinite(blc_ar_synop.lat.to_value(u.deg)),
            np.isfinite(trc_ar_synop.lon.to_value(u.deg)), np.isfinite(trc_ar_synop.lat.to_value(u.deg)))
    
    # Rotate bounding coordinates forward to the EIS time
    blc_ar_synop_rot = solar_rotate_coordinate(blc_ar_synop, time=eis_map.date)
    trc_ar_synop_rot = solar_rotate_coordinate(trc_ar_synop, time=eis_map.date)

    if debug:
        print("after solar_rotate:", blc_ar_synop_rot.lon, blc_ar_synop_rot.lat, "|", trc_ar_synop_rot.lon, trc_ar_synop_rot.lat)
        print("rotate finite:", np.isfinite(blc_ar_synop_rot.lon.to_value(u.deg)), np.isfinite(blc_ar_synop_rot.lat.to_value(u.deg)),
            np.isfinite(trc_ar_synop_rot.lon.to_value(u.deg)), np.isfinite(trc_ar_synop_rot.lat.to_value(u.deg)))

    # Select pixels that are either above or below the gauss values, these pixels will be used as seed points for PFSS fieldline tracing.
    masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data >=max_gauss) | (m_hmi_resample.data < min_gauss)) # np.where returns the (row, column) indices (masked_pix_y, masked_pix_x) of the selected pixels.

    print(f"Number of masked pixels: {len(masked_pix_x)}")
    print("Filtered Values:", np.nanmin(m_hmi_resample.data[masked_pix_y, masked_pix_x]), np.nanmax(m_hmi_resample.data[masked_pix_y, masked_pix_x]), np.nanmean(m_hmi_resample.data[masked_pix_y, masked_pix_x]))
    if debug:
        print("Filtered Values sample:", m_hmi_resample.data[masked_pix_y, masked_pix_x][:20])
    if debug:
        plt.hist(m_hmi_resample.data[masked_pix_y, masked_pix_x].flatten(), bins=50)
        plt.title("Histogram of Magnetic Field Strengths of Masked Pixels")
        plt.xlabel("Magnetic Field Strength (Gauss)")
        plt.ylabel("Number of Pixels")
        plt.grid(True)
        plt.show() # Checks the distribution of field strengths among selected seeds.

    # Convert the masked strong-field pixel positions (masked_pix_x, masked_pix_y) into real-world solar coordinates (longitude, latitude).
    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d() #.make_3d() converts the 2D pixel coordinates to 3D spherical coords.
    seeds_all = seeds
    print(f"[{tag}] Number of initial seed points: {len(seeds_all)}")
    if debug:
        print("seed lon range:", np.nanmin(seeds.lon.to_value(u.deg)), np.nanmax(seeds.lon.to_value(u.deg)))
        print("seed lat range:", np.nanmin(seeds.lat.to_value(u.deg)), np.nanmax(seeds.lat.to_value(u.deg)))

        plt.figure(figsize=(8,6))
        plt.scatter(seeds.lon.to(u.deg), seeds.lat.to(u.deg), s=1, c='r')
        plt.title("Initial Seeds Before FOV Filtering")
        plt.xlabel("Carrington Longitude (deg)")
        plt.ylabel("Carrington Latitude (deg)")

        plt.grid(True)
        plt.show() # Checks that strong-field seeds are planted everywhere there should be magnetic activity.

    if debug:
        print("blc lon/lat:", blc_ar_synop_rot.lon.to(u.deg), blc_ar_synop_rot.lat.to(u.deg))
        print("trc lon/lat:", trc_ar_synop_rot.lon.to(u.deg), trc_ar_synop_rot.lat.to(u.deg))
        print("lon min>max?", blc_ar_synop_rot.lon > trc_ar_synop_rot.lon)
        print("lat min>max?", blc_ar_synop_rot.lat > trc_ar_synop_rot.lat)
    if (not np.isfinite(blc_ar_synop_rot.lon.to_value(u.deg)) or
        not np.isfinite(blc_ar_synop_rot.lat.to_value(u.deg)) or
        not np.isfinite(trc_ar_synop_rot.lon.to_value(u.deg)) or
        not np.isfinite(trc_ar_synop_rot.lat.to_value(u.deg))):
        raise ValueError("NaN in rotated Carrington bounds (blc/trc).")

    in_lon = np.logical_and(seeds.lon > blc_ar_synop_rot.lon, seeds.lon < trc_ar_synop_rot.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop_rot.lat, seeds.lat < trc_ar_synop_rot.lat)

    # Filters based on the previous set HMI magnetogram FOV, only keeping the seeds that are within the FOV.
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]
    print(f"[{tag}] Number of seeds after FOV filtering: {len(seeds)}")

    if len(seeds) == 0:
        print(f"[{tag}] ZERO-SEED EVENT")
        print(f"[{tag}] EIS time: {Time(eis_map.date)}")
        print(f"[{tag}] HMI time: {Time(m_hmi_resample.date)}")
        print(f"[{tag}] blc/trc (Carr rot) lon/lat: {blc_ar_synop_rot.lon.to_value(u.deg)}, {blc_ar_synop_rot.lat.to_value(u.deg)} | {trc_ar_synop_rot.lon.to_value(u.deg)}, {trc_ar_synop_rot.lat.to_value(u.deg)}")
        print(f"[{tag}] seed lon range: {np.nanmin(seeds_all.lon.to_value(u.deg))} .. {np.nanmax(seeds_all.lon.to_value(u.deg))}")
        print(f"[{tag}] seed lat range: {np.nanmin(seeds_all.lat.to_value(u.deg))} .. {np.nanmax(seeds_all.lat.to_value(u.deg))}")
        return OpenFieldLines([]), ClosedFieldLines([])

    if debug:
        plt.figure(figsize=(8,6))
        plt.scatter(seeds.lon.to(u.deg), seeds.lat.to(u.deg), s=1, c='b')
        plt.title("Seeds After FOV Filtering (EIS area)")
        plt.xlabel("Carrington Longitude (deg)")
        plt.ylabel("Carrington Latitude (deg)")
        plt.legend()
        plt.grid(True)
        plt.show() # Confirm that after masking, seeds correspond only to the EIS field-of-view (plus 10% buffer).

    nrho = 70 # Number of radial grid points(steps form solar surface to source surface, like resolution).
    rss = 2.5  # Source surface radius (in solar radii, where the fieldlines are traced to, boundary condition for the model).
    pfss_input = pfsspy.Input(m_hmi_resample, nrho, rss) # .Input tell pfsspy what magentogram to use what radial grid to use and where to place the source surface.
    pfss_output = pfsspy.pfss(pfss_input) # .pfss solves the pfss problom and outputs a solution.

    ds = 0.01 # Step size for fieldline tracing (Each tracing step moves the fieldline by 0.01 R before recalculating direction).
    max_steps = int(np.ceil(10 * nrho / ds)) # .ceil rounds to the nearest integer, this computes a maximum number of steps that guarantees a fieldline can reach the top (2.5 R) or bottom (1 R) without runnin g out of steps.
    tracer = pfsspy.tracing.FortranTracer(step_size=ds, max_steps=max_steps) # Initialize a tracer to follow magnetic fieldlines step-by-step through the solved PFSS field.
    print('processing fieldlines')
    fieldlines = tracer.trace(SkyCoord(seeds), pfss_output) # .trace takes list of seed starting points, takes magentic field solution, tracing the fieldlines starting at each seed point. Still spherical coords.
    # Fieldline reaches the source surface (2.5) = open fieldline. Fieldline reaches the solar surface (1) = closed fieldline. The fieldline hits max_steps and is forcibly stopped.
    empty_coords_count = sum(len(f.coords) == 0 for f in fieldlines)
    print(f"Fieldlines with empty coords: {empty_coords_count} / {len(fieldlines)}")

    if debug:
        footpoints_lon = [f.coords.lon[0].to(u.deg).value for f in fieldlines if len(f.coords.lon) > 0]
        footpoints_lat = [f.coords.lat[0].to(u.deg).value for f in fieldlines if len(f.coords.lat) > 0]
    
        plt.figure(figsize=(8,6))
        plt.scatter(footpoints_lon, footpoints_lat, s=1, c='g')
        plt.title("Fieldline Starting Footpoints After Tracing")

        lon_min_rot = blc_ar_synop_rot.lon.deg
        lon_max_rot = trc_ar_synop_rot.lon.deg
        lat_min_rot = blc_ar_synop_rot.lat.deg
        lat_max_rot = trc_ar_synop_rot.lat.deg
        plt.gca().add_patch(Rectangle(
            (lon_min_rot, lat_min_rot),
            lon_max_rot - lon_min_rot,
            lat_max_rot - lat_min_rot,
            edgecolor='cyan',
            facecolor='none',
            lw=2,
            linestyle='--',
            label='Rotated EIS FOV'
        ))

        print(f"Δ Carrington Lon after rotation: {(trc_ar_synop_rot.lon - trc_ar_synop.lon).to(u.deg)}")
        print(f"Δ Carrington Lat after rotation: {(trc_ar_synop_rot.lat - trc_ar_synop.lat).to(u.deg)}")

        plt.grid(True)
        plt.show() # Confirms that fieldlines were successfully traced from the seeds.
    seeds = SkyCoord(seeds)
    seeds_2d = seeds.transform_to(eis_map.pixel_to_world(0*u.pix, 0*u.pix).frame)
    x_pix, y_pix = eis_map.world_to_pixel(seeds_2d)
    
    if debug:
        print("==========================")
        print("Adding seed metadata to fieldlines...")
        print("Seed frame before transformation:", seeds.frame.name)
        print("Map frame (EIS):", eis_map.coordinate_frame)
        print("Sample seed Carrington lon/lat before transform:", seeds[0].lon.deg, seeds[0].lat.deg)
        print("Sample seed Solar X Y after transform:", seeds_2d[0].Tx.to(u.arcsec), seeds_2d[0].Ty.to(u.arcsec))
        test_coord = seeds_2d[0]
        print("Seed 0 world coordinate (Tx, Ty):", test_coord.Tx.to(u.arcsec), test_coord.Ty.to(u.arcsec))
        print("Mapped to pixel:", eis_map.world_to_pixel(test_coord))
        print("===========================")
        print(f"x_pix range: {x_pix.min()} to {x_pix.max()}")
        print(f"y_pix range: {y_pix.min()} to {y_pix.max()}")
        print(f"Map shape (Y, X): {eis_map.data.shape}")
    
    x_vals = x_pix.value
    y_vals = y_pix.value
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    
    if debug:
        print(f"Skipped {np.count_nonzero(~valid)} fieldlines due to invalid pixel mapping.")
    valid_fieldlines = np.array(fieldlines)[valid] # Keep only fieldlines whose seeds mapped successfully to image pixels.
    for f, x, y in zip(valid_fieldlines, x_vals[valid], y_vals[valid]):

        # Loop length and loop starting pixel metadata
        f.start_pix = (int(x), int(y)) # Round to nearest pixel since image indices must be integers, not sub-pixel floats.

        coords = f.coords.cartesian # Converts from spherical to x, y, z, FortranTracer is done in spherical coords.
        # Extract the x, y, z coords:
        x = coords.x.value # .value removes the units so it is compatible with numpy, creating a float.
        y = coords.y.value
        z = coords.z.value

        points = []
        for i in range(len(x)):
            point = [x[i], y[i], z[i]] # creates a single point x_i, y_i, z_i
            points.append(point) # adds the point to the list of points.
        points_diff = [] # This will store the distance between each point and the next.
        for i in range(len(points)-1):
            dx = points[i+1][0] - points[i][0] # This grabs the x coord of the next point and subtracts the x coord of the current point.
            dy = points[i+1][1] - points[i][1]
            dz = points[i+1][2] - points[i][2]
            point_diff = [dx, dy, dz] # This creates a list of the differences in x, y, z.
            points_diff.append(point_diff) # This adds the point_diff to the list of points_diff.

        total_length = 0
        for i in range(len(points_diff)):
            points_diff_x_squared = points_diff[i][0] ** 2 # This grabs the x coord of the current point_diff, and squarses it.
            points_diff_y_squared = points_diff[i][1] ** 2 
            points_diff_z_squared = points_diff[i][2] ** 2 
            point_mag = np.sqrt(points_diff_x_squared + points_diff_y_squared + points_diff_z_squared)
            total_length += point_mag # This adds the absolute value of the point_diff to the total length of the fieldline.
        f.length = total_length
        
    for f in valid_fieldlines:
        '''
        ## pfss_output is the output of psspy.pfss()
        ## get_bvec() tells pfsspy to calcualte the magnetic field vector at the coords of the fieldline f.
        ## out_type="cartesian" means we get the output in cartesian coords, isntead of spherical.
        #bvec = pfss_output.get_bvec(f.coords, out_type="cartesian") * u.G
        #f.b = bvec # does nothing.
        ## np.linalg.norm calculates the length(magnitude) of the vector bvec, which is the magnetic field vector in this case.
        ## axis=1 means operate within rows (across columns).
        ## .value removes the units
        ## .mean gets the average of all the values.
        #f.mean_B = np.mean(np.linalg.norm(bvec.value, axis=1))
        '''

        coords = f.coords # f.coords is a list of 3D coord points along the fieldline f.
        if len(coords) == 0:
            f.mean_B = np.nan
            continue
        #bvec_unitless = pfss_output.get_bvec(coords, out_type="cartesian") # This gets the magnetic field vector at the coords of the fieldline f.
        #bvec = bvec_unitless * u.G # This converts the units from Tesla to Gauss.
        coords.representation_type = "spherical" # Makes sure the coord representation is in spherical form.
        #print("Max radius after filtering:", np.max(coords.radius.to(u.R_sun).value))
        phi = coords.lon.to("rad").value # Extracts the longitude of the coords in radians.
        sin_theta = np.sin(coords.lat).value # Extracts the sine of the latitude of the coords.
        log_r = np.log(coords.radius.to(u.R_sun).value) # Extracts the log of the radius of the coords in solar radii.
        N = len(phi)
        interp_input = np.zeros((N, 3))  # create empty (N, 3) array.
        for i in range(N):
            interp_input[i, 0] = phi[i]         # φ (longitude, in radians)
            interp_input[i, 1] = sin_theta[i]   # sin(θ)
            interp_input[i, 2] = log_r[i]       # log(r)
        bvec_unitless = pfss_output._brgi(interp_input) # Use PFSSPy's internal interpolator _brgi to get the B-vector at each coord point.
        unit_str = pfss_output.input_map.meta.get("bunit", None) # Attempt to get the unit string from the metadata of the input map.
        bunit = u.Unit(unit_str) if unit_str is not None else u.dimensionless_unscaled # In our case the bunit is unitless.
        bvec = bvec_unitless * bunit
        bvec = bvec * u.G # This converts the units from Tesla to Gauss (works because we know bunit in this case is unitless, but we use anyway to stay consistent with the original function).
        bvec_mag = []
        for i in range(len(bvec)):
            bvec_x = bvec[i][0].value # This grabs the x coord of the magnetic field vector.
            bvec_y = bvec[i][1].value 
            bvec_z = bvec[i][2].value 
            mag = np.sqrt(bvec_x ** 2 + bvec_y ** 2 + bvec_z ** 2) # This calculates the magnitude of the magnetic field vector.
            bvec_mag.append(mag)
        #bvec_mean = np.mean(bvec_mag) # This takes the average of all |B| values along the fieldline. If any value is NaN, the mean will be NaN.
        bvec_mean = np.nanmean(bvec_mag) # This takes the average of all |B| values along the fieldline, ignoring NaN values.
        f.mean_B = bvec_mean # This adds the mean magnetic field strength to the fieldline object.

        '''
        if not hasattr(f, "custom"):
            f.custom = {}

        f.custom["expansion_factor"] = f.expansion_factor if f.expansion_factor is not None else np.nan # This adds the expansion factor to the fieldline object, if it exists. If the expansion factor is None, set it to NaN.
        #print(f"Expansion factor: {f.expansion_factor}")
        #print(f"Custom stored expansion factor: {f.custom['expansion_factor']}")

        Expansion factor : a measure of how much the magnetic field expands from the solar surface to the source surface.
        First thing we need to do is find out r0 and B0, which are the values at the footpoint of the fieldline (1 R).
        
        r_values = coords.radius.to(u.R_sun).value # Extract the radius for each point along the fieldline.
        smallest_diff = float('inf')  # Start with a huge difference.
        r_closest_to_1 = None
        B_at_r_closest_to_1 = None
        r_at_r_closest_to_1 = None
        for i in range(len(r_values)):
           r_curr = r_values[i]
           B_curr = bvec_mag[i]
           diff_from_1 = abs(r_curr - 1.0) # This calculates the difference between the current radius and 1.0.
           if diff_from_1 < smallest_diff: # This checks if the current difference is smaller than the smallest difference.
               smallest_diff = diff_from_1
               r_closest_to_1 = r_curr
               B_at_r_closest_to_1 = B_curr
               r_at_r_closest_to_1 = r_curr
        B0 = B_at_r_closest_to_1 
        r0 = r_at_r_closest_to_1

        # Next we need to find out r1 and B1, which are the values at the source surface (2.5 R).
        r_target = 2.5
        smallest_diff = float('inf')
        r_closest_to_2_5 = None
        B_at_r_closest_to_2_5 = None
        for i in range(len(r_values)):
           r_curr = r_values[i]
           B_curr = bvec_mag[i]
           diff_from_2_5 = abs(r_curr - r_target)
           if diff_from_2_5 < smallest_diff: # This checks if the current difference is smaller than the smallest difference.
               smallest_diff = diff_from_2_5
               r_closest_to_2_5 = r_curr
               B_at_r_closest_to_2_5 = B_curr
        B1 = B_at_r_closest_to_2_5
        r1 = r_closest_to_2_5

        if B1 > 0 and r0 > 0 and r1 > 0:
           f_expansion = (B0 / B1) * (r1 / r0)**2
        else:
           f_expansion = np.nan
        if not hasattr(f, "custom"):
           f.custom = {}
        f.custom["expansion_factor"] = f_expansion # Not a predefined attribute in pfsspy, but we can add it as a custom attribute. 
        '''
    
    num_with_length = sum(np.isfinite(f.length) for f in valid_fieldlines)
    print(f"Fieldlines with valid length metadata: {num_with_length} / {len(valid_fieldlines)}")

    num_with_mean_B = sum(np.isfinite(f.mean_B) for f in valid_fieldlines)
    print(f"Fieldlines with valid mean_B metadata: {num_with_mean_B} / {len(valid_fieldlines)}")

    if debug:
        # Plot to verify that start_pix aligns with EIS data.
        plt.figure(figsize=(6, 10))
        plt.imshow(eis_map.data, origin='lower', cmap='gray', aspect='auto')
        x_pix = [f.start_pix[0] for f in valid_fieldlines]
        y_pix = [f.start_pix[1] for f in valid_fieldlines]
        plt.scatter(x_pix, y_pix, s=2, color='cyan', label='start_pix')
        plt.title("EIS Raster with Fieldline Start Pixels (start_pix)")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.legend()
        plt.grid(True)
        plt.show() # Shows whether the fieldlines actually map back to EIS data in the correct orientation.

    if debug:
        x_check, y_check = eis_map.world_to_pixel(eis_map.pixel_to_world(0*u.pix, 0*u.pix))
        print(f"Pixel (0,0) round trip lands at: ({x_check}, {y_check})")
        
        plt.figure()
        plt.hist([f.mean_B for f in valid_fieldlines if np.isfinite(f.mean_B)], bins=50)
        plt.title("Distribution of Mean Magnetic Field Strengths")
        plt.xlabel("Mean |B| (Gauss)")
        plt.ylabel("Number of Fieldlines")
        plt.grid(True)
        plt.show() # Checks the distribution of mean magnetic field strengths among the fieldlines.

    open_lines = [f for f in fieldlines if f.is_open] # For each fieldline f in fieldlines, check if f.is_open == True, if yes add to open_lines
    closed_lines = [f for f in fieldlines if not f.is_open] # For each fieldline f in fieldlines, check if f.is_open == False, if yes add to closed_lines
    open_fieldlines = OpenFieldLines(open_lines) if open_lines else OpenFieldLines([]) # If open_lines is not empty, create OpenFieldLines object, else create an empty one.
    closed_fieldlines = ClosedFieldLines(closed_lines) if closed_lines else ClosedFieldLines([]) # If closed_lines is not empty, create ClosedFieldLines object, else create an empty one.

    print(f"Total field lines: {len(fieldlines)}")
    print(f"Open field lines: {len(open_fieldlines)}")
    print(f"Closed field lines: {len(closed_fieldlines)}")
    return open_fieldlines, closed_fieldlines
        
if __name__ == '__main__':
    import os 
    IRIS_map_dirs = sorted(glob('/Users/andysh.to/Script/Data/IRIS_output/201904*/*_v_turb_map.fits'))
    for iris_map_dir in IRIS_map_dirs:
        name = iris_map_dir.replace('_v_turb_map.fits','_pfss_fieldlines.pickle')
        # if os.path.exists(name):
        #     print(f'{name.split("/")[-1]} exists... Skipping...')
        # else:
        try:
            get_pfss(iris_map_dir)
        except Exception as e:
            print(f'Error: {e}')
            print(f'{name.split("/")[-1]} did not process...')

            continue

    
