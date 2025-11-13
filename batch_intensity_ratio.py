import sunpy.map
import os
import re
import pickle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from astropy.coordinates import SkyCoord
import glob
from astropy import units as u
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import multiprocessing
import matplotlib.patches as mpatches
from skimage.filters import gaussian
from sunpy.coordinates.frames import Helioprojective
from sunpy.map import Map
from astropy.visualization import ImageNormalize
from astropy.io import fits
import eispac
from astropy.visualization import ImageNormalize
import matplotlib.patches as mpatches 
from matplotlib.patches import Rectangle
from iris_get_pfss_utils import hmi_daily_download
import matplotlib.patches as mpatches 
import scipy.ndimage
from matplotlib.colors import Normalize
from astropy.constants import R_sun
from astropy.io import fits
import pprint
from scipy.stats import linregress
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from iris_get_pfss_utils import hmi_daily_download, get_pfss_from_map
import asheis
from pathlib import Path
from matplotlib import colors
from sunpy.map import Map
from reproject import reproject_interp
import h5py
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as imshift
#from eispac.instr import ccd_offset
import argparse  # for -c/--cores


# Intensity ratio + CCD fix + scalling for G(T)

intens_dir = Path("intensity_map")
ratio_out_dir = Path("intensity_ratio")
ratio_out_dir.mkdir(parents=True, exist_ok=True)
alias = {
    "ca_14_193.87": "ca14193_87",
    "ar_14_194.40": "ar14194_40",
    "si_10_258.37": "si10258_37",
    "s_10_264.23":  "s10264_23",
    "fe_16_262.98": "fe16262_98",
    "s_13_256.69":  "s13256_69",
    "s_11_188.68":  "s11188_68",
    "ar_11_188.81": "ar11188_81",
    "fe_12_196.64": "fe12196_64",
    "s_10_196.81":  "s10196_81",
    "fe_13_196.53": "fe13196_53",
    "ar_11_194.10": "ar11194_10",
    "s_11_191.26":  "s11191_26",
}


# Hardcoded timestamp for the dataset
def _discover_timestamps(intens_dir: Path) -> list[str]:
    """
    Scan intensity_map/*.fits and return a sorted list of unique timestamps,
    e.g. '2012_06_10__12_30_19'. Robust to any line suffix after the time.
    """
    ts = set()
    for p in intens_dir.glob("*.fits"):
        stem = p.stem  # e.g., '2012_06_10__12_30_19_ar11188_81'
        if "__" not in stem:
            continue
        head, tail = stem.split("__", 1)                # '2012_06_10', '12_30_19_ar...'
        time_bits = tail.split("_")[:3]                 # ['12','30','19']
        if len(time_bits) == 3:
            ts.add(f"{head}__{'_'.join(time_bits)}")    # '2012_06_10__12_30_19'
    return sorted(ts)

timestamps = _discover_timestamps(intens_dir)
print(f"Discovered {len(timestamps)} timestamps under {intens_dir}:")
for t in timestamps:
    print("  ", t)


def _elem_symbol(line_id: str) -> str:
    return line_id.split('_', 1)[0].capitalize()

def _pair_labels(e1: str, e2: str):
    pretty = f"{_elem_symbol(e1)} / {_elem_symbol(e2)}"   # e.g., "Si / S"
    safe   = f"{_elem_symbol(e1)}_{_elem_symbol(e2)}"     # e.g., "Si_S"
    return pretty, safe
def _wav(line_id: str) -> float:
    # e.g. 'fe_16_262.98' -> 262.98
    return float(line_id.split('_')[-1])
def _align_to_ref(img, ref):
    """
    Align 'img' to 'ref' (same shape) using subpixel cross-correlation.
    Returns aligned_img, (dy, dx), error.
    """
    mask = np.isfinite(img) & np.isfinite(ref)
    img0 = np.where(mask, img, 0.0)
    ref0 = np.where(mask, ref, 0.0)
    (dy, dx), error, _ = phase_cross_correlation(ref0, img0, upsample_factor=10)
    aligned = imshift(img, shift=(dy, dx), order=1, mode="nearest", prefilter=False)
    return aligned, (dy, dx), error


# reference plasma point used in the paper (provenance only)
pair_ref = {
    ('ca_14_193.87', 'ar_14_194.40'): {'logt0': 6.6,  'ne0': 1e9},
    ('fe_16_262.98', 's_13_256.69'):  {'logt0': 6.4,  'ne0': 1e9},
    ('si_10_258.37', 's_10_264.23'):  {'logt0': 6.2,  'ne0': 1e9},
    ('s_11_188.68',  'ar_11_188.81'): {'logt0': 6.25, 'ne0': 1e9},
}

# theory ratio r_th_ref (num/den) read from the g(t) plots at (logt0, ne0)
# fill these with the values
pair_r = {
    ('ca_14_193.87', 'ar_14_194.40'): 1.3856,  # ca/ar @ logt=6.6, ne=1e9
    ('fe_16_262.98', 's_13_256.69'):  0.2591,  # fe/s  @ logt=6.4, ne=1e9
    ('si_10_258.37', 's_10_264.23'):  2.29,  # si/s  @ logt=6.2, ne=1e9
    ('s_11_188.68',  'ar_11_188.81'): 1.7010,  # s/ar  @ logt=6.25, ne=1e9
}


def plot_composition_map(timestamp, element1, element2):
    # Output FITS/PNG paths
    pretty_pair, safe_pair = _pair_labels(element1, element2)
    fits_filename = str(ratio_out_dir / f"intensity_map_ratio_{timestamp}_{safe_pair}.fits")
    png_filename  = ratio_out_dir / f"intensity_map_ratio_{timestamp}_{safe_pair}.png"
    # Load the two per-line intensity FITS
    num_path = intens_dir / f"{timestamp}_{alias[element1]}.fits"
    den_path = intens_dir / f"{timestamp}_{alias[element2]}.fits"
    if not num_path.exists() or not den_path.exists():
        print(f"[SKIP] {timestamp} missing inputs for {element1}/{element2}:")
        print(f"num: {num_path}  exists={num_path.exists()}")
        print(f"den: {den_path}  exists={den_path.exists()}")
        return

    # Error maps saved by batch_intensity_maps.py alongside intensities
    err_num_path = intens_dir / f"{timestamp}_{alias[element1]}_err.fits"
    err_den_path = intens_dir / f"{timestamp}_{alias[element2]}_err.fits"
    if not err_num_path.exists() or not err_den_path.exists():
        print(f"[SKIP] {timestamp} missing error FITS for {element1}/{element2}:")
        print(f"       num_err: {err_num_path}  exists={err_num_path.exists()}")
        print(f"       den_err: {err_den_path}  exists={err_den_path.exists()}")
        return

    m_num = Map(str(num_path))
    m_den = Map(str(den_path))
    m_err_num = Map(str(err_num_path))
    m_err_den  = Map(str(err_den_path))

    print(f"[{timestamp}] {element1} unit:", m_num.meta.get("BUNIT"),
          "| exptime:", m_num.meta.get("EXPTIME"))
    print(f"[{timestamp}] {element2} unit:", m_den.meta.get("BUNIT"),
          "| exptime:", m_den.meta.get("EXPTIME"))

    fe12_path = Path(f"aligned_fe12_intensity_maps/aligned_eis_{timestamp}_intensity.fits")
    if not fe12_path.exists():
        print(f"[SKIP] Missing Fe XII aligned map for {timestamp}: {fe12_path}")
        return
    fe12_map = Map(str(fe12_path))




    num_reproj, _ = reproject_interp(m_num, fe12_map.wcs, shape_out=fe12_map.data.shape)
    den_reproj, _ = reproject_interp(m_den, fe12_map.wcs, shape_out=fe12_map.data.shape)
    err_num_reproj, _ = reproject_interp(m_err_num, fe12_map.wcs, shape_out=fe12_map.data.shape)
    err_den_reproj, _ = reproject_interp(m_err_den, fe12_map.wcs, shape_out=fe12_map.data.shape)


    '''
    Asheis actually already does offset to headers in batch_intensity_maps.py
    '''
    # CCD offset already applied in WCS by asheis; use reprojected arrays as-is
    num = num_reproj.astype(float)
    den = den_reproj.astype(float)
    err_num = err_num_reproj.astype(float)
    err_den = err_den_reproj.astype(float)

    # CCD offset handling (to match asheis) 
    # asheis does NOT estimate a shift from the image data (no phase-correlation).
    # Instead, it applies a fixed, wavelength-dependent CCD Y-offset from the
    # instrument model (eispac.instr.ccd_offset), referenced to Fe XII 195.119 Å.
    # In asheis this is applied by adjusting the map/WCS; here we implement the same
    # idea by shifting the reprojected arrays along Solar-Y by dy(λ).
    # Why use the model instead of data-driven phase correlation?
    # - Stable across rasters; not fooled by low-SNR/patchy lines (e.g., Fe XVI 262.98).
    # - Preserves numerator/denominator co-registration for same-CCD pairs.

    #wave_num = float(element1.split('_')[-1])   # e.g. 'fe_16_262.98' -> 262.98
    #wave_den = float(element2.split('_')[-1])
    
    ## Model CCD offsets relative to Fe XII 195.119 Å (may be scalar OR 2-vector)
    #off_ref = np.asarray(ccd_offset(195.119*u.AA).to_value('pixel'), dtype=float)
    #off_num = np.asarray(ccd_offset(wave_num*u.AA).to_value('pixel'), dtype=float)
    #off_den = np.asarray(ccd_offset(wave_den*u.AA).to_value('pixel'), dtype=float)


    #delta_num = off_ref - off_num   # desired shift for numerator: (dy, dx) or scalar
    #delta_den = off_ref - off_den   # desired shift for denominator
    
    ## Normalize to (dy, dx) floats (don’t use absolute; sign matters)
    #if delta_num.ndim == 0:
    #    dy_num, dx_num = float(delta_num), 0.0
    #else:
    #    dy_num = float(delta_num[0])
    #    dx_num = float(delta_num[1]) if delta_num.size > 1 else 0.0
    
    #if delta_den.ndim == 0:
    #    dy_den, dx_den = float(delta_den), 0.0
    #else:
    #    dy_den = float(delta_den[0])
    #    dx_den = float(delta_den[1]) if delta_den.size > 1 else 0.0
    

    ## Shift along image axes: (axis-0, axis-1) == (dy, dx)
    #num = imshift(num_reproj, shift=(dy_num, dx_num), order=1, mode="nearest", prefilter=False).astype(float)
    #den = imshift(den_reproj, shift=(dy_den, dx_den), order=1, mode="nearest", prefilter=False).astype(float)
    
    #print(f"{timestamp} {element1}/{element2} model CCD Δ(dy,dx) num=({dy_num:.2f},{dx_num:.2f})px "
    #      f"den=({dy_den:.2f},{dx_den:.2f})px")
    
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
    ratio = np.full(fe12_map.data.shape, np.nan, dtype=float)
    ratio[valid] = num[valid] / den[valid]

    # NEW: theory-based normalization using the paper's G(T) reference 
    norm_used = "none"
    norm_factor = 1.0
    # 1) read theory ratio R at the reference (logT0, ne0)
    pair = (element1, element2)
    ref_info = pair_ref.get(pair, {'logt0': np.nan, 'ne0': 1e9})
    r_theory = pair_r[pair]  # theory ratio at (logt0, ne0) from the plot
    # 2) compute S = 1/R and apply
    norm_factor = 1.0 / r_theory
    ratio *= norm_factor
    # σ_R = |R| * sqrt( (σ_num/num)^2 + (σ_den/den)^2 )
    ratio_err = np.full(fe12_map.data.shape, np.nan, dtype=float)
    ratio_err[valid] = np.abs(ratio[valid]) * np.sqrt(
        (err_num[valid] / num[valid])**2 + (err_den[valid] / den[valid])**2
    )
    # 3) record provenance
    norm_used = f"theory(logt0={ref_info['logt0']}, ne0={ref_info['ne0']})"


    # Build ratio map on Fe XII WCS/meta
    ratio_map = Map(ratio, fe12_map.meta.copy())
    ratio_map.meta.pop('bunit', None)
    ratio_map.meta['measurement'] = f"{element1}/{element2}"
    ratio_map.meta['ratio_norm'] = norm_used
    ratio_map.meta['ratio_norm_factor'] = norm_factor
    ratio_map.meta['ratio_ref_logt0'] = ref_info['logt0']
    ratio_map.meta['ratio_ref_ne0']   = ref_info['ne0']
    ratio_map.meta['ratio_ref_r']     = r_theory  # store the r you read from the plot
    fits_err_filename = str(ratio_out_dir / f"intensity_map_ratio_{timestamp}_{safe_pair}_err.fits")
    ratio_err_map = Map(ratio_err, fe12_map.meta.copy())
    ratio_err_map.meta.pop('bunit', None)
    ratio_err_map.meta['measurement'] = f"{element1}/{element2} (error)"
    ratio_err_map.meta['ratio_norm'] = norm_used
    ratio_err_map.meta['ratio_norm_factor'] = norm_factor
    ratio_err_map.save(fits_err_filename, overwrite=True)

    # Fixed ranges: S/Ar vs others
    is_s_ar = (element1.startswith("s_11_") and element2.startswith("ar_11_"))
    if is_s_ar:
        cmap = "RdBu_r"; vmin, vmax = 0.5, 1.5
    else:
        cmap = "viridis"; vmin, vmax = 1.0, 4.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Save FITS
    ratio_map.save(fits_filename, overwrite=True)

    # Plot & save PNG
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ratio_map.wcs)
    title_text = f"{pretty_pair} intensity ratio\n{timestamp}"
    im = ratio_map.plot(title=title_text, norm=norm, cmap=cmap, axes=ax)
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.1)
    plt.savefig(png_filename, dpi=150)  # matplotlib accepts Path directly
    plt.close()
    print(f"Saved: {fits_filename}")
    print(f"Saved: {png_filename}")

ratios = [
    ('ca_14_193.87', 'ar_14_194.40'),  # Ca XIV / Ar XIV
    ('fe_16_262.98', 's_13_256.69'),   # Fe XVI / S XIII
    ('si_10_258.37', 's_10_264.23'),   # Si X / S X   
    ('s_11_188.68',  'ar_11_188.81'),  # S XI / Ar XI
]

def _work(task):
    ts, e1, e2 = task
    try:
        return plot_composition_map(ts, e1, e2)
    except FileNotFoundError as e:
        print(e)
        return None


#if __name__ == "__main__":
#    for timestamp in timestamps:
#        for elem1, elem2 in ratios:
#            try:
#                plot_composition_map(timestamp, elem1, elem2)
#            except FileNotFoundError as e:
#                print(e)

if __name__ == "__main__":
    # (A) parse cores like the other script
    parser = argparse.ArgumentParser(description="Make composition ratio maps")
    parser.add_argument("-c", "--cores", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    # (B) build the job list (cartesian product of timestamps × ratios)
    jobs = [(ts, e1, e2) for ts in timestamps for (e1, e2) in ratios]

    # (C) parallel map, Andy-style: simple Pool with processes=args.cores
    if args.cores == 1:
        for task in tqdm(jobs, total=len(jobs)):
            _work(task)
    else:
        with multiprocessing.Pool(processes=args.cores) as pool:   # <— added line
            for _ in tqdm(pool.imap_unordered(_work, jobs, chunksize=1), total=len(jobs)):
                pass
