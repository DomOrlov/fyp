import sunpy.map 
from sunpy.net import Fido, attrs as a
from astropy import units as u
from tqdm import tqdm
from sunkit_image.coalignment import _calculate_shift as calculate_shift
import os
from astropy.io import fits
import glob
from pathlib import Path
import argparse  # for -c/--cores
import multiprocessing  # for Pool
from astropy.coordinates import SkyCoord      # WHY: submap needs coords in AIA frame
import numpy as np                            # WHY: z-normalization before xcorr
from sunpy.coordinates import SphericalScreen  # WHY: avoid issues near limb

# File to log non-aligned files
non_aligned_log = Path("non_aligned_files.txt")

# Enable test mode
test_mode = False
test_file = "eis_2014_02_02__14_19_52_intensity.fits"

def alignment(eis_fit, return_shift=False, wavelength=193 * u.angstrom):
    """
    Aligns the EIS map with the AIA map by calculating the shift in coordinates 
    and applying the shift to the EIS map using cross-correlation.

    Parameters:
    eis_fit (str): The file path of the EIS map FITS file.
    
    Returns:
    sunpy.map.Map: The aligned EIS map.
    """
    
    fe12_directory = Path("nonaligned_fe12_intensity_maps")     # raw Fe XII from maker
    aligned_fe12_directory = Path("aligned_fe12_intensity_maps") # aligner output
    aia_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data").resolve()  # AIA 


    # Construct the full file path
    eis_fit_path = fe12_directory / eis_fit
    out_path = Path("aligned_fe12_intensity_maps") / f"aligned_{eis_fit}"
    if out_path.exists():
        print(f"[SKIP] already aligned: {out_path.name}")
        return


    # Load the EIS map with error handling for corrupt FITS files
    try:
        eis_map_int = sunpy.map.Map(eis_fit_path)
    except Exception as e:
        print(f"Error loading EIS map for {eis_fit_path}: {e}")
        with open(non_aligned_log.as_posix(), 'a') as log_file:
            log_file.write(f"{eis_fit} - Failed to load EIS map: {e}\n")
        return

    # Load the corresponding Fe XII 195.12 Å map for header extraction
    #fe12_filename = "_".join(eis_fit.split('_')[:7]) + "_intensity.fits"
    # Use the input filename directly instead
    fe12_filename = eis_fit
    fe12_fit_path = fe12_directory / fe12_filename


    try:
        fe12_map = sunpy.map.Map(fe12_fit_path)
        header_fe12 = fe12_map.meta
        print(f"Loaded Fe XII 195.12 Å header from {fe12_fit_path}")
    except Exception as e:
        print(f"Error loading Fe XII map for {fe12_fit_path}: {e}")
        with open(non_aligned_log.as_posix(), 'a') as log_file:
            log_file.write(f"{fe12_fit_path} - Failed to load Fe XII map: {e}\n")
        return

    # Extract date and time (up to minutes) from the EIS filename
    date_part = "_".join(eis_fit.split('_')[1:4])  # YYYY_MM_DD
    time_part = eis_fit.split('__')[1].split('_')[0] + "_" + eis_fit.split('__')[1].split('_')[1]

    # Use glob to match any file with the same date and time (HH_MM), ignoring seconds/milliseconds
    pattern = str((aia_dir / f"aia.lev1.193A_{date_part}T{time_part}_*.image_lev1.fits"))
    # Find matching files locally
    matching_files = glob.glob(pattern)

    # Output the search pattern and matching files for debugging
    print(f"Search pattern used: {pattern}")
    print(f"Matching files found: {matching_files}")

    # Check if the file already exists locally
    if matching_files:
        local_aia_path = matching_files[0]
        print(f"Found existing AIA file locally: {local_aia_path}")
        aia_map = sunpy.map.Map(local_aia_path)
    else:
        print(f"Failed to find matching local AIA files. Search pattern was: {pattern}")
        # Search for AIA map within a specific time range and wavelength
        aia_result = Fido.search(
            a.Time(eis_map_int.date - 5 * u.second, eis_map_int.date + 10 * u.second),
            a.Instrument('AIA'), 
            a.Wavelength(193 * u.angstrom), 
            a.Sample(1 * u.minute)
        )
        
        # Fetch the AIA map and save it to a temporary directory
        try:
            fetched_files = Fido.fetch(
                aia_result, 
                path=aia_dir,  # Use absolute path
                overwrite=False
            )    
            if not fetched_files:
                print(f"Warning: No AIA data found for {eis_fit}. Skipping this file.")
                return

            aia_map = sunpy.map.Map(fetched_files[0])
        except Exception as e:
            print(f"Error fetching AIA data for {eis_fit}: {e}")
            return












    # --- Crop AIA to the EIS FoV (+ margin) so xcorr sees the same scene
    # WHY: Matching scene removes unrelated features that break xcorr.
    #eis_bl = fe12_map.bottom_left_coord.transform_to(aia_map.coordinate_frame)
    #eis_tr = fe12_map.top_right_coord.transform_to(aia_map.coordinate_frame)



    #with SphericalScreen(frame=aia_map.coordinate_frame):
    #    eis_bl = fe12_map.bottom_left_coord.transform_to(aia_map.coordinate_frame)
    #    eis_tr = fe12_map.top_right_coord.transform_to(aia_map.coordinate_frame)

    # Use a spherical screen so off-disk EIS corners don't turn into NaNs when
    # transforming into the AIA frame. Center it on the AIA observer.
    with SphericalScreen(aia_map.observer_coordinate, only_off_disk=False):
        eis_bl = fe12_map.bottom_left_coord.transform_to(aia_map.coordinate_frame)
        eis_tr = fe12_map.top_right_coord.transform_to(aia_map.coordinate_frame)


    # DEBUG: check EIS corners in AIA frame and ordering
    print(f"[{eis_fit}] EIS->AIA corners:"
        f" BL(Tx,Ty)=({eis_bl.Tx.to(u.arcsec).value:.1f},{eis_bl.Ty.to(u.arcsec).value:.1f}),"
        f" TR(Tx,Ty)=({eis_tr.Tx.to(u.arcsec).value:.1f},{eis_tr.Ty.to(u.arcsec).value:.1f})")

    # If ordering is flipped in either axis, fix it (submap expects BL<TopRight)
    Tx_min = np.minimum(eis_bl.Tx, eis_tr.Tx)
    Tx_max = np.maximum(eis_bl.Tx, eis_tr.Tx)
    Ty_min = np.minimum(eis_bl.Ty, eis_tr.Ty)
    Ty_max = np.maximum(eis_bl.Ty, eis_tr.Ty)






    margin = 50 * u.arcsec  # give xcorr a little context without letting other ARs dominate
    #blm = SkyCoord(eis_bl.Tx - margin, eis_bl.Ty - margin, frame=aia_map.coordinate_frame)
    #trm = SkyCoord(eis_tr.Tx + margin, eis_tr.Ty + margin, frame=aia_map.coordinate_frame)
    blm = SkyCoord(Tx_min - margin, Ty_min - margin, frame=aia_map.coordinate_frame)
    trm = SkyCoord(Tx_max + margin, Ty_max + margin, frame=aia_map.coordinate_frame)

    try:
        aia_crop = aia_map.submap(blm, top_right=trm)


        print(f"[{eis_fit}] AIA crop shape: {getattr(aia_crop.data, 'shape', None)}")
        if aia_crop.data.size == 0 or aia_crop.data.shape[0] == 0 or aia_crop.data.shape[1] == 0:
            print(f"[SKIP] empty AIA crop for {eis_fit}")
            with open(non_aligned_log.as_posix(), 'a') as f:
                f.write(f"{eis_fit} - Empty AIA crop\n")
            return



    except Exception as e:
        print(f"[WARN] AIA submap failed ({e}); continuing un-cropped (less robust)")
        aia_crop = aia_map


    ## Calculate the resampling factors for aligning the maps
    #n_x = (aia_map.scale.axis1 * aia_map.dimensions.x) / eis_map_int.scale.axis1
    #n_y = (aia_map.scale.axis2 * aia_map.dimensions.y) / eis_map_int.scale.axis2
    
    ## Resample the AIA map
    #aia_map_r = aia_map.resample(u.Quantity([n_x, n_y]))
    # --- Match pixel geometry for xcorr: same shape as EIS
    ny, nx = fe12_map.data.shape
    aia_map_r = aia_crop.resample(u.Quantity([nx, ny], u.pixel))
    print(f"[{eis_fit}] EIS shape: {fe12_map.data.shape}, AIA resampled shape: {aia_map_r.data.shape}")


    if aia_map_r.data.shape != fe12_map.data.shape or aia_map_r.data.size == 0:
        print(f"[SKIP] bad resample for {eis_fit}")
        with open(non_aligned_log.as_posix(), 'a') as f:
            f.write(f"{eis_fit} - Bad resample (shape {aia_map_r.data.shape} vs {fe12_map.data.shape})\n")
        return




    # --- Z-normalize arrays for a cleaner correlation peak
    def _znorm(a):
        a = np.asarray(a, dtype=np.float64)
        a -= np.nanmean(a)
        s = np.nanstd(a)
        return a / s if s > 0 else a

    A = _znorm(aia_map_r.data)
    B = _znorm(fe12_map.data)
    A[~np.isfinite(A)] = 0.0
    B[~np.isfinite(B)] = 0.0





    print(f"[{eis_fit}] finite A/B: {np.isfinite(A).sum()}/{np.isfinite(B).sum()}, "
        f"std A/B: {np.nanstd(A):.3g}/{np.nanstd(B):.3g}")

    if not np.isfinite(A).any() or not np.isfinite(B).any():
        print(f"[SKIP] no finite pixels {eis_fit}")
        with open(non_aligned_log.as_posix(), 'a') as f:
            f.write(f"{eis_fit} - No finite pixels after z-norm\n")
        return

    # extremely uniform images produce useless correlation
    if np.nanstd(A) < 1e-9 or np.nanstd(B) < 1e-9:
        print(f"[SKIP] near-constant image {eis_fit}")
        with open(non_aligned_log.as_posix(), 'a') as f:
            f.write(f"{eis_fit} - Near-constant image (std too small)\n")
        return


    # Calculate the shift in coordinates between the AIA and EIS maps
    #yshift, xshift = calculate_shift(aia_map_r.data, fe12_map.data)

    #yshift_pix, xshift_pix = calculate_shift(A, B)  # returns pixel shifts (A→B)



    try:
        yshift_pix, xshift_pix = calculate_shift(A, B)
    except Exception as e:
        print(f"[WARN] xcorr failed for {eis_fit}: {e}")
        with open(non_aligned_log.as_posix(), 'a') as f:
            f.write(f"{eis_fit} - xcorr failed: {e}\n")
        return





    # Convert the shift in coordinates to world coordinates
    #reference_coord = aia_map_r.pixel_to_world(xshift, yshift)
    #Txshift = reference_coord.Tx - fe12_map.bottom_left_coord.Tx
    #Tyshift = reference_coord.Ty - fe12_map.bottom_left_coord.Ty
    # --- Pixel → world conversion using resampled AIA scale
    Txshift = xshift_pix * aia_map_r.scale.axis1
    Tyshift = yshift_pix * aia_map_r.scale.axis2

    #print(eis_map_int.date)
    #print(f"px shift: (x,y)=({xshift_pix:.2f},{yshift_pix:.2f})")
    #print(f"arcsec shift: (Tx,Ty)=({Txshift.to(u.arcsec):.2f},{Tyshift.to(u.arcsec):.2f})")


    
    # Print the date and shift values for debugging
    print(eis_map_int.date)
    print(f"Date: {eis_map_int.date}, Txshift: {Txshift}, Tyshift: {Tyshift}")
    print(f"Shift in arcsec: |Tx| = {abs(Txshift.to(u.arcsec).value)}, |Ty| = {abs(Tyshift.to(u.arcsec).value)}")

    # Check if the shift is within a certain range
    #if (abs(Tyshift / u.arcsec) < 150) and (abs(Txshift / u.arcsec) < 150):
    #    aligned_fe12_map = fe12_map.shift_reference_coord(Txshift, Tyshift)
    #    print(f'shifted - Tx:{Txshift}, Ty:{Tyshift}')
    #else:
    #    aligned_fe12_map = fe12_map
    #    print(f'not shifted - Tx:{Txshift}, Ty:{Tyshift}')
    #    with open(non_aligned_log, 'a') as log_file:
    #        log_file.write(
    #            f"{eis_fit} - Not shifted, "
    #            f"Tx: {Txshift}, Ty: {Tyshift}, "
    #            f"|Tx| (arcsec): {abs(Txshift.to(u.arcsec).value)}, "
    #            f"|Ty| (arcsec): {abs(Tyshift.to(u.arcsec).value)}\n"
    #        )
    max_abs = 150 * u.arcsec  # WHY: reject obviously bad correlations
    if (abs(Txshift) < max_abs) and (abs(Tyshift) < max_abs):
        aligned_fe12_map = fe12_map.shift_reference_coord(-Txshift, -Tyshift)
        print(f"shifted - Tx:{Txshift}, Ty:{Tyshift}")
    else:
        aligned_fe12_map = fe12_map
        print(f"not shifted - Tx:{Txshift}, Ty:{Tyshift}")
        with open(non_aligned_log, 'a') as log_file:
            log_file.write(
                f"{eis_fit} - Not shifted, "
                f"Tx: {Txshift}, Ty: {Tyshift}, "
                f"|Tx| (arcsec): {abs(Txshift.to(u.arcsec).value)}, "
                f"|Ty| (arcsec): {abs(Tyshift.to(u.arcsec).value)}\n"
            )


    # Apply the Fe XII header to the hacked map
    aligned_fe12_map.meta.update(header_fe12)


    # Define the output file path for the aligned map
    output_path = aligned_fe12_directory / f"aligned_{eis_fit}"
    aligned_fe12_map.save(output_path.as_posix(), overwrite=True)
    print(f"Saved aligned map to {output_path}")

def _work(eis_fit):
    # one-arg wrapper so Pool can call alignment on each filename
    return alignment(eis_fit)

# Test mode to only process the specified file
fe12_directory = Path("nonaligned_fe12_intensity_maps")

if test_mode:
    eis_files = [test_file]
else:
    eis_files = sorted([p.name for p in fe12_directory.glob("eis_*_intensity.fits")])



#for num, fit in tqdm(enumerate(eis_files), total=len(eis_files)):
#    alignment(fit)

#print(f"Non-aligned files have been logged to {non_aligned_log}")
if __name__ == "__main__":
    # (A) parse cores like your other scripts
    parser = argparse.ArgumentParser(description="Align EIS Fe XII maps to AIA 193")
    parser.add_argument("-c", "--cores", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    # (B) job list = the filenames we discovered (already strings)
    jobs = eis_files

    # (C) Andy-style pooling
    if args.cores == 1:
        for fit in tqdm(jobs, total=len(jobs)):
            _work(fit)
    else:
        with multiprocessing.Pool(processes=args.cores) as pool:
            for _ in tqdm(pool.imap_unordered(_work, jobs, chunksize=1), total=len(jobs)):
                pass

    print(f"Non-aligned files have been logged to {non_aligned_log}")
