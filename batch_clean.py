# Cleaning intensity ratio via masking and Feature Detection 

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
import glob
import multiprocessing
import re
import argparse
import logging
import numpy as np
from astropy.io import fits
from sunpy.map import Map
from scipy.ndimage import label
logging.getLogger("sunpy").setLevel(logging.WARNING)

fyp_dir=  '/home/ug/orlovsd2/fyp'
data_dir = "/mnt/scratch/data/orlovsd2/sunpy/data"
base_dir = "/mnt/scratch/data/orlovsd2/sunpy/data/intensity_ratio"
intensity_dir = "/mnt/scratch/data/orlovsd2/sunpy/data/intensity_map"
aligned_dir = "/mnt/scratch/data/orlovsd2/sunpy/data/aligned_fe12_intensity_maps"
fits_files = sorted(glob.glob(f"{base_dir}/*.fits"))

test_mode = True
test_target = "2014_02_05__10_41_27"


# maps for numerator/denominator intensity file suffixes
abundance_to_intensity_map = {
    "CaAr": "ca14193_87",
    "FeS":  "fe16262_98",
    "sis":  "si10258_37",
    "sar":  "s11188_68",
}
abundance_to_denom_map = {
    "CaAr": "ar14194_40",
    "FeS":  "s13256_69",
    "sis":  "s10264_23",
    "sar":  "ar11188_81",
}

relerr = 0.20
min_feature_size = 75    

def eis_stamp_from_timestamp(ts: str) -> str:
    # "2014_02_01__10_50_35" -> "eis_20140201_105035"
    d = ts.replace("_","")
    return f"eis_{d[:8]}_{ts.split('__')[1].replace('_','')}"

def parse_ratio_name(fname: str):
    """
    intensity_map_ratio_<YYYY>_<MM>_<DD>__<HH>_<MM>_<SS>_<Token>.fits
    where <Token> is 'Fe_S', 'Ca_Ar', 'Si_S', or 'S_Ar'.
    """
    m = re.match(r"intensity_map_ratio_(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})_([A-Za-z]+_[A-Za-z]+)\.fits$", fname)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def normalize_pair_token(pair_token: str) -> str:
    # map file token -> dict key
    return (pair_token
            .replace("Fe_S","FeS")
            .replace("Ca_Ar","CaAr")
            .replace("Si_S","sis")
            .replace("S_Ar","sar"))

def _work(fits_file):
    base_name = os.path.basename(fits_file)
    if not base_name.endswith(".fits") or base_name.startswith("cleaned_relerr_size_"):
        return

    input_fits = fits_file
    output_fits = os.path.join(base_dir, "cleaned_relerr_size_" + base_name)
    if os.path.exists(output_fits):
        return


    date_time_key, pair_token = parse_ratio_name(base_name)
    if date_time_key is None:
        return

    if test_mode and test_target not in base_name:
        return

    element_abbrev = normalize_pair_token(pair_token)

    num_key = abundance_to_intensity_map[element_abbrev]
    den_key = abundance_to_denom_map[element_abbrev]

    # load ratio as the thing we will mask
    aligned_header_path = os.path.join(aligned_dir, f"aligned_eis_{date_time_key}_intensity.fits")
    aligned_header = Map(aligned_header_path, silence_warnings=True).meta

    ratio_data = fits.getdata(input_fits).astype(float)
    ratio_map  = Map(ratio_data, aligned_header)
    ratio_arr  = ratio_map.data.copy()

    original_valid = int(np.isfinite(ratio_arr).sum())
    total_pixels   = ratio_arr.size

    # load intensities (numerator & denominator) and convert to DN/s
    num_path = os.path.join(intensity_dir, f"{date_time_key}_{num_key}.fits")
    den_path = os.path.join(intensity_dir, f"{date_time_key}_{den_key}.fits")

    num_map_raw = Map(num_path, silence_warnings=True)
    den_map_raw = Map(den_path, silence_warnings=True)


    num_err_path = os.path.join(intensity_dir, f"{date_time_key}_{num_key}_err.fits")
    den_err_path = os.path.join(intensity_dir, f"{date_time_key}_{den_key}_err.fits")
    
    num_err_map = Map(num_err_path, silence_warnings=True)
    den_err_map = Map(den_err_path, silence_warnings=True)

    num_int = num_map_raw.data.astype(float)
    den_int = den_map_raw.data.astype(float)
    num_er = num_err_map.data.astype(float)
    den_er = den_err_map.data.astype(float)
    
    # Valid pixels are those where intensity and error are finite and intensity > 0
    good_num = np.isfinite(num_int) & np.isfinite(num_er) & (num_int > 0)
    good_den = np.isfinite(den_int) & np.isfinite(den_er) & (den_int > 0)
    
    num_relerr = np.full_like(num_int, np.nan, dtype=float)
    den_relerr = np.full_like(den_int, np.nan, dtype=float)
    num_relerr[good_num] = num_er[good_num] / num_int[good_num]
    den_relerr[good_den] = den_er[good_den] / den_int[good_den]
    
    pass_num = good_num & (num_relerr <= relerr)
    pass_den = good_den & (den_relerr <= relerr)
    
    keep_mask = pass_num & pass_den
    ratio_arr[~keep_mask] = np.nan
    after_relerr_valid = int(np.isfinite(ratio_arr).sum())
    valid_mask = np.isfinite(ratio_arr)

    
    labels, nlab = label(valid_mask)
    sizes = np.bincount(labels.ravel())
    
    keep_lab = sizes >= min_feature_size   # keep only features >= min_feature_size pixels
    keep_lab[0] = False                    # label 0 is not a feature
    
    ratio_arr[~keep_lab[labels]] = np.nan
    after_feature_valid = int(np.isfinite(ratio_arr).sum())

    print(f"\n{base_name}")
    print(f"Total pixels: {total_pixels}")
    print(f"Valid before: {original_valid} ({original_valid/total_pixels:.2%})")
    print(f"After relerr (<= {relerr:.2f}): {after_relerr_valid} ({after_relerr_valid/total_pixels:.2%})")
    print(f"Feature check: before ={after_relerr_valid} ({after_relerr_valid/total_pixels:.2%})")
    print(f"after= {after_feature_valid} ({after_feature_valid/total_pixels:.2%})")

    Map(ratio_arr, aligned_header).save(output_fits, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean intensity ratio maps")
    parser.add_argument("-c", "--cores", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    jobs = fits_files

    if args.cores == 1:
        for fits_file in jobs:
            _work(fits_file)
    else:
        # with multiprocessing.Pool(processes=args.cores) as pool:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.cores, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_work, jobs, chunksize=1):
                pass
