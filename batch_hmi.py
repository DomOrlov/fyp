# automated pickle test_target = "2014_02_05__10_41_27"
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
import glob
import pickle
import sunpy.map
from iris_get_pfss_utils import get_pfss_from_map
from pathlib import Path
import argparse  # for -c/--cores
import multiprocessing  # for Pool

data_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data").resolve()
pickle_dir = data_dir / "pfss_pickles"
Path(pickle_dir).mkdir(parents=True, exist_ok=True)

full_geometry = False
test_mode = False
test_target = "2014_02_05__10_41_27"

fe12_dir = data_dir / "aligned_fe12_intensity_maps"
fits_files = sorted(glob.glob(f"{fe12_dir}/*.fits"))

#for fits_file in fits_files:
#    print(f"Processing: {fits_file}")
#    base_name = os.path.basename(fits_file).replace("aligned_", "").replace("_intensity.fits", "")

#    if test_mode and test_target not in base_name:
#        continue  # Skip all files except the test_target
    
#    open_pickle_filename = f"{pickle_dir}/{base_name}_open_fieldlines.pickle"
#    closed_pickle_filename = f"{pickle_dir}/{base_name}_closed_fieldlines.pickle"
    
#    if os.path.isfile(open_pickle_filename) and os.path.isfile(closed_pickle_filename):
#        print(f"Skipping {base_name} (pickles already exist).")
#        continue  # Skip to the next file

#    eis_map = sunpy.map.Map(fits_file) #sunpy.map specialises in handeling 2D solar images, .Map takesin input and makes a proper solar image object
    
#    open_fieldlines, closed_fieldlines = get_pfss_from_map(eis_map, min_gauss=-5, max_gauss=5, dimension=(1080, 540))

#    print(f"Saving {len(open_fieldlines)} open field lines to {open_pickle_filename}")
#    print(f"Saving {len(closed_fieldlines)} closed field lines to {closed_pickle_filename}")
    

#    with open(open_pickle_filename, 'wb') as f:
#        pickle.dump(open_fieldlines, f)

#    with open(closed_pickle_filename, 'wb') as f:
#        pickle.dump(closed_fieldlines, f)

def _work(fits_file):
    base_name = os.path.basename(fits_file).replace("aligned_", "").replace("_intensity.fits", "")

    if test_mode and test_target not in base_name:
        return  

    open_pickle_filename = f"{pickle_dir}/{base_name}_open_fieldlines.pickle"
    closed_pickle_filename = f"{pickle_dir}/{base_name}_closed_fieldlines.pickle"

    if os.path.isfile(open_pickle_filename) and os.path.isfile(closed_pickle_filename):
        print(f"Skipping {base_name} (pickles already exist).")
        return
      
    print(f"Processing: {fits_file}")
    eis_map = sunpy.map.Map(fits_file)

    open_fieldlines, closed_fieldlines = get_pfss_from_map(eis_map, min_gauss=-5, max_gauss=5, dimension=(1080, 540))
    if not full_geometry:
        open_fieldlines = [
            (int(f.start_pix[0]), int(f.start_pix[1]), float(f.length), float(f.mean_B))
            for f in open_fieldlines
            if hasattr(f, "start_pix") and hasattr(f, "length") and hasattr(f, "mean_B")
        ]

        closed_fieldlines = [
            (int(f.start_pix[0]), int(f.start_pix[1]), float(f.length), float(f.mean_B))
            for f in closed_fieldlines
            if hasattr(f, "start_pix") and hasattr(f, "length") and hasattr(f, "mean_B")
        ]



    print(f"Saving {len(open_fieldlines)} open field lines to {open_pickle_filename}")
    print(f"Saving {len(closed_fieldlines)} closed field lines to {closed_pickle_filename}")

    with open(open_pickle_filename, 'wb') as f:
        pickle.dump(open_fieldlines, f)

    with open(closed_pickle_filename, 'wb') as f:
        pickle.dump(closed_fieldlines, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace PFSS field lines and write pickles")
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
