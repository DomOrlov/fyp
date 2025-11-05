# automated pickle test_target = "2014_02_05__10_41_27"
import os
import glob
import pickle
import sunpy.map
from iris_get_pfss_utils import get_pfss_from_map
from pathlib import Path

iris_dir = "pfss_pickles"
Path(iris_dir).mkdir(parents=True, exist_ok=True)

test_mode = True
test_target = "2014_02_05__10_41_27"

fe12_dir = "aligned_fe12_intensity_maps"
fits_files = sorted(glob.glob(f"{fe12_dir}/*.fits"))

for fits_file in fits_files:
    print(f"Processing: {fits_file}")
    base_name = os.path.basename(fits_file).replace("aligned_", "").replace("_intensity.fits", "")

    if test_mode and test_target not in base_name:
        continue  # Skip all files except the test_target
    
    open_pickle_filename = f"{iris_dir}/{base_name}_open_fieldlines.pickle"
    closed_pickle_filename = f"{iris_dir}/{base_name}_closed_fieldlines.pickle"
    
    if os.path.isfile(open_pickle_filename) and os.path.isfile(closed_pickle_filename):
        print(f"Skipping {base_name} (pickles already exist).")
        continue  # Skip to the next file

    eis_map = sunpy.map.Map(fits_file) #sunpy.map specialises in handeling 2D solar images, .Map takesin input and makes a proper solar image object
    
    open_fieldlines, closed_fieldlines = get_pfss_from_map(eis_map, min_gauss=-5, max_gauss=5, dimension=(1080, 540))

    print(f"Saving {len(open_fieldlines)} open field lines to {open_pickle_filename}")
    print(f"Saving {len(closed_fieldlines)} closed field lines to {closed_pickle_filename}")
    

    with open(open_pickle_filename, 'wb') as f:
        pickle.dump(open_fieldlines, f)

    with open(closed_pickle_filename, 'wb') as f:
        pickle.dump(closed_fieldlines, f)