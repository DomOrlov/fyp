#loop length pixel aligned fe12 to get a hect
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
import glob
import pickle
import numpy as np
import sunpy.map
import astropy.units as u
from pathlib import Path
import argparse  # for -c/--cores
import multiprocessing



data_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data").resolve()
pickle_files = sorted(glob.glob(f"{data_dir}/pfss_pickles/*_closed_fieldlines.pickle"))


#for closed_pickle_path in pickle_files:
#    #date_time = "_".join(os.path.basename(closed_pickle_path).split("_")[2:9])
#    #fe12_path = f"{data_dir}/aligned_fe12_intensity_maps/aligned_eis_{date_time}_intensity.fits"
#    date_time = os.path.basename(closed_pickle_path)[len("eis_"):-len("_closed_fieldlines.pickle")]
#    fe12_path = f"{data_dir}/aligned_fe12_intensity_maps/aligned_eis_{date_time}_intensity.fits"

#    print(f"\nProcessing date_time = {date_time}")
#    print(f"Closed fieldline pickle: {closed_pickle_path}")
#    print(f"FE XII path: {fe12_path}")
#    # print(f"  Output FITS path: {output_fits_path}")

#    if not os.path.exists(fe12_path):
#        print(f"Missing FE XII file for {date_time}, skipping.")
#        continue

#    fe12_map = sunpy.map.Map(fe12_path) # Load the aligned FE XII intensity map as a SunPy map
#    print("Pixel (0,0) Solar-Y:", fe12_map.pixel_to_world(0 * u.pixel, 0 * u.pixel).Ty)
#    print("Pixel (0,511) Solar-Y:", fe12_map.pixel_to_world(0 * u.pixel, 511 * u.pixel).Ty)

#    meta = fe12_map.meta  # Use SunPy map metadata directly

    
#    with open(closed_pickle_path, 'rb') as f:
#        closed_fieldlines = pickle.load(f)
#    open_pickle_path = closed_pickle_path.replace("_closed_fieldlines.pickle", "_open_fieldlines.pickle")
#    with open(open_pickle_path, "rb") as f:
#        open_fieldlines = pickle.load(f)

#    ny, nx = fe12_map.data.shape  

#    # Create blank maps
#    # HECT outputs (single pass)
#    loop_map_closed = np.full((ny, nx), np.nan) # km
#    loop_map_open = np.full((ny, nx), np.nan) # km
#    mean_B_map_combined = np.full((ny, nx), np.nan) # Gauss
#    open_mask_map = np.full((ny, nx), np.nan) # 1=open, 0=closed, nan=unused

	

#    # Closed fieldlines: fill length (km) and mean_B (G), mark mask=0
#    for f in closed_fieldlines:
#        if hasattr(f, "start_pix"):
#            x, y = f.start_pix
#            if 0 <= x < nx and 0 <= y < ny:
#                if hasattr(f, "length"):
#                    length_km = f.length / 1e3
#                    if np.isfinite(length_km) and (length_km > 0):
#                        loop_map_closed[y, x] = length_km
#                if hasattr(f, "mean_B"):
#                    mean_B = f.mean_B
#                    if np.isfinite(mean_B) and (mean_B > 0):
#                        mean_B_map_combined[y, x] = mean_B
#                        open_mask_map[y, x] = 0



#    excluded_count = 0

#    # Open fieldlines
#    for f in open_fieldlines:
#        if hasattr(f, "start_pix"):
#            x, y = f.start_pix
#            length_km = f.length / 1e3 if hasattr(f, "length") else np.nan

#            if 0 <= x < nx and 0 <= y < ny:
#                if np.isfinite(length_km) and (length_km > 0):
#                    loop_map_open[y, x] = length_km
#                if hasattr(f, "mean_B"):
#                    mean_B = f.mean_B
#                    if np.isfinite(mean_B) and (mean_B > 0):
#                        mean_B_map_combined[y, x] = mean_B
#                        open_mask_map[y, x] = 1
#            else:
#                excluded_count += 1

#    print(f"Total open fieldlines excluded due to being outside FE XII FOV: {excluded_count}")

#    loop_map_closed_smap = sunpy.map.Map(loop_map_closed, meta)
#    loop_map_closed_smap.meta["BUNIT"] = "km"
#    loop_map_closed_smap.save(f"{data_dir}/loop_length/loop_length_map_closed_{date_time}.fits", overwrite=True)
    
#    loop_map_open_smap = sunpy.map.Map(loop_map_open, meta)
#    loop_map_open_smap.meta["BUNIT"] = "km"
#    loop_map_open_smap.save(f"{data_dir}/loop_length/loop_length_map_open_{date_time}.fits", overwrite=True)

#    n_valid_closed = np.count_nonzero(np.isfinite(loop_map_closed))
#    n_valid_open = np.count_nonzero(np.isfinite(loop_map_open))
#    print(f"Closed map valid pixels: {n_valid_closed}")
#    print(f"Open map valid pixels: {n_valid_open}")

#    solar_y_seed_values = [fe12_map.pixel_to_world(x * u.pixel, y * u.pixel).Ty.to(u.arcsec).value
#                           for f in closed_fieldlines if hasattr(f, "start_pix")
#                           for x, y in [f.start_pix]]
#    print("Seed Solar-Y range:", np.min(solar_y_seed_values), "to", np.max(solar_y_seed_values))

#    print(f"Total closed fieldlines: {len(closed_fieldlines)}")
#    print(f"Fieldlines with length > 0: {sum(f.length > 0 for f in closed_fieldlines if hasattr(f, 'length'))}")

#    print(f"Total open fieldlines: {len(open_fieldlines)}")
#    print(f"Fieldlines with length > 0: {sum(f.length > 0 for f in open_fieldlines if hasattr(f, 'length'))}")

#    for f in open_fieldlines:
#        if not hasattr(f, "length") or f.length <= 0 or len(f.coords) <= 1:
#            print(f"Degenerate open fieldline — length: {getattr(f, 'length', 'N/A')}, coords: {len(getattr(f, 'coords', []))}")


#    print(f"Saved closed loop length map to: loop_length_map_closed_{date_time}.fits")
#    print(f"Saved open loop length map to: loop_length_map_open_{date_time}.fits")

#    # Save mean-B combined and open/closed mask
#    mean_B_smap = sunpy.map.Map(mean_B_map_combined, meta)
#    mean_B_smap.meta["BUNIT"] = "Gauss"
#    mean_B_smap.save(f"{data_dir}/mean_B/mean_B_map_combined_{date_time}.fits", overwrite=True)

#    mask_smap = sunpy.map.Map(open_mask_map, meta)
#    mask_smap.meta["BUNIT"] = "1"
#    mask_smap.save(f"{data_dir}/mean_B/mean_B_open_mask_map_{date_time}.fits", overwrite=True)

#    print(f"Saved closed/open loop-length, mean-B, and mask for {date_time}")

def _work(closed_pickle_path):
    # Extract date_time and paths
    date_time = os.path.basename(closed_pickle_path)[len("eis_"):-len("_closed_fieldlines.pickle")]
    fe12_path = f"{data_dir}/aligned_fe12_intensity_maps/aligned_eis_{date_time}_intensity.fits"

    print(f"\nProcessing date_time = {date_time}")
    print(f"Closed fieldline pickle: {closed_pickle_path}")
    print(f"FE XII path: {fe12_path}")

    # Check FE XII availability
    if not os.path.exists(fe12_path):
        print(f"Missing FE XII file for {date_time}, skipping.")
        return

    # Load FE XII map
    fe12_map = sunpy.map.Map(fe12_path)
    print("Pixel (0,0) Solar-Y:", fe12_map.pixel_to_world(0 * u.pixel, 0 * u.pixel).Ty)
    print("Pixel (0,511) Solar-Y:", fe12_map.pixel_to_world(0 * u.pixel, 511 * u.pixel).Ty)
    meta = fe12_map.meta

    # Load pickles
    with open(closed_pickle_path, 'rb') as f:
        closed_fieldlines = pickle.load(f)
    open_pickle_path = closed_pickle_path.replace("_closed_fieldlines.pickle", "_open_fieldlines.pickle")
    with open(open_pickle_path, "rb") as f:
        open_fieldlines = pickle.load(f)

    # Allocate outputs
    ny, nx = fe12_map.data.shape
    loop_map_closed = np.full((ny, nx), np.nan) # km
    loop_map_open = np.full((ny, nx), np.nan) # km
    mean_B_map_combined = np.full((ny, nx), np.nan) # Gauss
    open_mask_map = np.full((ny, nx), np.nan) # 1=open, 0=closed, nan=unused

    # Closed fieldlines

    # for f in closed_fieldlines:
    #     if hasattr(f, "start_pix"):
    #         x, y = f.start_pix
    #         if 0 <= x < nx and 0 <= y < ny:
    #             if hasattr(f, "length"):
    #                 length_km = f.length / 1e3
    #                 if np.isfinite(length_km) and (length_km > 0):
    #                     loop_map_closed[y, x] = length_km
    #             if hasattr(f, "mean_B"):
    #                 mean_B = f.mean_B
    #                 if np.isfinite(mean_B) and (mean_B > 0):
    #                     mean_B_map_combined[y, x] = mean_B
    #                     open_mask_map[y, x] = 0

    for item in closed_fieldlines:
        # tuple case (from full_geometry=False)
        if isinstance(item, tuple):
            x, y, length_m, mean_B = item

            if 0 <= x < nx and 0 <= y < ny:
                length_km = length_m / 1e3
                if np.isfinite(length_km) and (length_km > 0):
                    loop_map_closed[y, x] = length_km

                if np.isfinite(mean_B) and (mean_B > 0):
                    mean_B_map_combined[y, x] = mean_B
                    open_mask_map[y, x] = 0

        # Full geometry case
        else:
            f = item
            if hasattr(f, "start_pix"):
                x, y = f.start_pix
                if 0 <= x < nx and 0 <= y < ny:
                    if hasattr(f, "length"):
                        length_km = f.length / 1e3
                        if np.isfinite(length_km) and (length_km > 0):
                            loop_map_closed[y, x] = length_km
                    if hasattr(f, "mean_B"):
                        mean_B = f.mean_B
                        if np.isfinite(mean_B) and (mean_B > 0):
                            mean_B_map_combined[y, x] = mean_B
                            open_mask_map[y, x] = 0

    # Open fieldlines

    # excluded_count = 0
    # for f in open_fieldlines:
    #     if hasattr(f, "start_pix"):
    #         x, y = f.start_pix
    #         length_km = f.length / 1e3 if hasattr(f, "length") else np.nan
    #         if 0 <= x < nx and 0 <= y < ny:
    #             if np.isfinite(length_km) and (length_km > 0):
    #                 loop_map_open[y, x] = length_km
    #             if hasattr(f, "mean_B"):
    #                 mean_B = f.mean_B
    #                 if np.isfinite(mean_B) and (mean_B > 0):
    #                     mean_B_map_combined[y, x] = mean_B
    #                     open_mask_map[y, x] = 1
    #         else:
    #             excluded_count += 1

    excluded_count = 0
    for item in open_fieldlines:
        if isinstance(item, tuple):
            x, y, length_m, mean_B = item
            if 0 <= x < nx and 0 <= y < ny:
                length_km = length_m / 1e3
                if np.isfinite(length_km) and (length_km > 0):
                    loop_map_open[y, x] = length_km
                if np.isfinite(mean_B) and (mean_B > 0):
                    mean_B_map_combined[y, x] = mean_B
                    open_mask_map[y, x] = 1
            else:
                excluded_count += 1
        else:
            f = item
            if hasattr(f, "start_pix"):
                x, y = f.start_pix
                length_km = f.length / 1e3 if hasattr(f, "length") else np.nan
                if 0 <= x < nx and 0 <= y < ny:
                    if np.isfinite(length_km) and (length_km > 0):
                        loop_map_open[y, x] = length_km
                    if hasattr(f, "mean_B"):
                        mean_B = f.mean_B
                        if np.isfinite(mean_B) and (mean_B > 0):
                            mean_B_map_combined[y, x] = mean_B
                            open_mask_map[y, x] = 1
                else:
                    excluded_count += 1


    print(f"Total open fieldlines excluded due to being outside FE XII FOV: {excluded_count}")

    # Save loop-length maps
    loop_map_closed_smap = sunpy.map.Map(loop_map_closed, meta)
    loop_map_closed_smap.meta["BUNIT"] = "km"
    loop_map_closed_smap.save(f"{data_dir}/loop_length/loop_length_map_closed_{date_time}.fits", overwrite=True)

    loop_map_open_smap = sunpy.map.Map(loop_map_open, meta)
    loop_map_open_smap.meta["BUNIT"] = "km"
    loop_map_open_smap.save(f"{data_dir}/loop_length/loop_length_map_open_{date_time}.fits", overwrite=True)

    # Stats
    n_valid_closed = np.count_nonzero(np.isfinite(loop_map_closed))
    n_valid_open = np.count_nonzero(np.isfinite(loop_map_open))
    print(f"Closed map valid pixels: {n_valid_closed}")
    print(f"Open map valid pixels: {n_valid_open}")

    # solar_y_seed_values = [fe12_map.pixel_to_world(x * u.pixel, y * u.pixel).Ty.to(u.arcsec).value
    #                        for f in closed_fieldlines if hasattr(f, "start_pix")
    #                        for x, y in [f.start_pix]]

    solar_y_seed_values = []
    for item in closed_fieldlines:
        if isinstance(item, tuple):
            x, y = item[0], item[1]
        else:
            if not hasattr(item, "start_pix"):
                continue
            x, y = item.start_pix

        solar_y_seed_values.append(
            fe12_map.pixel_to_world(x * u.pixel, y * u.pixel).Ty.to(u.arcsec).value
        )


    if len(solar_y_seed_values) > 0:
        print("Seed Solar-Y range:", np.min(solar_y_seed_values), "to", np.max(solar_y_seed_values))

    print(f"Total closed fieldlines: {len(closed_fieldlines)}")
    # print(f"Fieldlines with length > 0: {sum(f.length > 0 for f in closed_fieldlines if hasattr(f, 'length'))}")
    n_len_pos_closed = 0
    for item in closed_fieldlines:
        if isinstance(item, tuple):
            length_m = item[2]
        else:
            length_m = item.length if hasattr(item, "length") else np.nan
        if np.isfinite(length_m) and length_m > 0:
            n_len_pos_closed += 1
    print(f"Fieldlines with length > 0: {n_len_pos_closed}")

    print(f"Total open fieldlines: {len(open_fieldlines)}")
    # print(f"Fieldlines with length > 0: {sum(f.length > 0 for f in open_fieldlines if hasattr(f, 'length'))}")
    n_len_pos_open = 0
    for item in open_fieldlines:
        if isinstance(item, tuple):
            length_m = item[2]
        else:
            length_m = item.length if hasattr(item, "length") else np.nan
        if np.isfinite(length_m) and length_m > 0:
            n_len_pos_open += 1
    print(f"Fieldlines with length > 0: {n_len_pos_open}")

    print(f"Saved closed loop length map to: loop_length_map_closed_{date_time}.fits")
    print(f"Saved open loop length map to: loop_length_map_open_{date_time}.fits")

    # Save mean-B combined and open/closed mask
    mean_B_smap = sunpy.map.Map(mean_B_map_combined, meta)
    mean_B_smap.meta["BUNIT"] = "Gauss"
    mean_B_smap.save(f"{data_dir}/mean_B/mean_B_map_combined_{date_time}.fits", overwrite=True)

    mask_smap = sunpy.map.Map(open_mask_map, meta)
    mask_smap.meta["BUNIT"] = "1"
    mask_smap.save(f"{data_dir}/mean_B/mean_B_open_mask_map_{date_time}.fits", overwrite=True)

    print(f"Saved closed/open loop-length, mean-B, and mask for {date_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build loop-length and mean-B maps from PFSS pickles")
    parser.add_argument("-c", "--cores", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    jobs = pickle_files

    if args.cores == 1:
        for p in jobs:
            _work(p)
    else:
        with multiprocessing.Pool(processes=args.cores) as pool:
            for _ in pool.imap_unordered(_work, jobs, chunksize=1):
                pass