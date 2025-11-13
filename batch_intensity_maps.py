#!/usr/bin/env python
import argparse
import numpy as np
import os
from ashmcmc import ashmcmc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
from sunpy.map import Map
from astropy.io import fits
import re

data_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data").resolve()
#fe12_dir = Path("nonaligned_fe12_intensity_maps")  # raw Fe XII from ashmcmc
fe12_dir = data_dir / "fe12_intensity_maps"
#aligned_fe12_dir = Path("aligned_fe12_intensity_maps") # produced by your aligner later
aligned_fe12_dir = data_dir / "aligned_fe12_intensity_maps"
#custom_intensity_dir = "intensity_map"
custom_intensity_dir = data_dir / "intensity_map"


def save_error_map_like(int_map: Map, err_array, out_path: Path):
    """
    Save an error map to FITS using the same WCS/header as int_map.
    """
    h = int_map.meta.copy()
    h["BUNIT"] = h.get("BUNIT", "") + " (error)"
    fits.writeto(out_path, data=err_array.astype(float), header=fits.Header(h), overwrite=True)



def list_eis_data_files(data_dir: Path) -> list[Path]:
    """
    Recursively find Hinode/EIS spectral cubes (*.data.h5), ignoring *.head.h5.
    """
    return sorted(data_dir.rglob("eis_*.data.h5"))


# robust timestamp parser (safer than .stem/.split)
_TS_RE = re.compile(r"eis_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.data\.h5$")

def eis_filename_to_timestamp(file_path: Path) -> str:
    """
    'eis_20150706_043512.data.h5' -> '2015_07_06__04_35_12'
    """
    m = _TS_RE.search(file_path.name)
    if not m:
        raise ValueError(f"Unrecognised EIS filename: {file_path.name}")
    Y, M, D, h, m, s = m.groups()
    return f"{Y}_{M}_{D}__{h}_{m}_{s}"

def make_intensity_maps_for_file(filename, line_databases, ncpu=4, test_mode=False):
    """
    Generate intensity maps for all lines in `line_databases`, or just one in test mode.
    """
    a = ashmcmc(filename, ncpu=ncpu)
    print(f"DEBUG: ashmcmc outdir => {a.outdir}")

    if test_mode:
        #line_databases = {"CaAr": ["ca_14_193.87"], "FeS": ["fe_16_262.98"]}
        #line_databases = {"FeS": ["s_13_256.69"]}  # only S XIII 256.69
        line_databases = {
            "CaAr": ["ca_14_193.87", "ar_14_194.40"],  # Ca XIV and Ar XIV
            "FeS":  ["fe_16_262.98", "s_13_256.69"]    # Fe XVI and S XIII
        }

    # --- Fe XII 195.12 (raw) for this file ---
    timestamp = eis_filename_to_timestamp(Path(filename))
    fe12_line = "fe_12_195.12"
    fe12_out = fe12_dir / f"eis_{timestamp}_intensity.fits"

    if fe12_out.exists():
        print(f"[SKIP] Fe XII already exists: {fe12_out.name}")
    else:
        try:
            print(f"\n--- Generating Fe XII {fe12_line} for {filename} ---")
            m_fe = a.ash.get_intensity(
                fe12_line,
                outdir=str(fe12_dir),   # ashmcmc’s own outputs; we still save a canonical filename below
                refit=False,
                plot=True,
                mcmc=False,
                calib=True,
                calib_year="2014"
            )
            # Defensive save under our canonical name
            from sunpy.map import Map as _Map
            _Map(m_fe.data, m_fe.meta).save(fe12_out.as_posix(), overwrite=True)
            print(f"Saved Fe XII: {fe12_out}")
        except Exception as e:
            print(f"Fe XII failed for {filename}: {e}")


    for ratio_key, ratio_lines in line_databases.items():
        for line in ratio_lines[:2]:
            print(f"\n--- Generating intensity map for {line} in {filename} ---")
            datetime_str = eis_filename_to_timestamp(Path(filename))

            # map alias exactly as your ratio step expects
            alias = {
                "ca_14_193.87": "ca14193_87",
                "ar_14_194.40": "ar14194_40",
                "si_10_258.37": "si10258_37",
                "s_10_264.23":  "s10264_23",
                "fe_16_262.98": "fe16262_98",
                "s_13_256.69":  "s13256_69",
                "s_11_188.68":  "s11188_68",
                "ar_11_188.81": "ar11188_81",
            }
            #out_fits = Path(custom_intensity_dir) / f"{datetime_str}_{alias.get(line, line.replace('.', '_'))}.fits"
            #if out_fits.exists():
            #    print(f"[SKIP] already exists: {out_fits.name}")
            #    continue
            out_fits = Path(custom_intensity_dir) / f"{datetime_str}_{alias.get(line, line.replace('.', '_'))}.fits"
            err_fits = out_fits.with_name(out_fits.stem + "_err.fits")

            # Skip only if BOTH intensity and error already exist
            if out_fits.exists() and err_fits.exists():
                print(f"[SKIP] already exists: {out_fits.name} (+err)")
                continue
            try:
                #m = a.ash.get_intensity(
                #    line,
                #    outdir=custom_intensity_dir,  
                #    refit=False,
                #    plot=True,
                #    mcmc=False,
                #    calib=True,
                #    calib_year="2014"
                #)
                # First: get arrays with per-pixel uncertainty (this performs the fit once)
                I_data, I_err = a.ash.get_intensity(
                    line,
                    outdir=custom_intensity_dir,  
                    refit=(not out_fits.exists() or not err_fits.exists()),  # << only refit if needed
                    plot=True,
                    mcmc=True,      # returns (I, sigma_I) when we refit
                    calib=True,
                    calib_year="2014"
                )


                # Second: cheaply load a Map (header/WCS) without refitting or plotting
                m = a.ash.get_intensity(
                    line,
                    outdir=custom_intensity_dir,
                    refit=False,
                    plot=False,
                    mcmc=False,         # returns a Map object
                    calib=True,
                    calib_year="2014"
                )

                #print(f"DEBUG: Intensity Stats for {line} -> Min={m.data.min()}, Max={m.data.max()}, Mean={m.data.mean()}")
                #print(f"DEBUG: Nonzero pixel count for {line}: {m.data.nonzero()[0].size}")
                # Force-save FITS file
                # === Convert line (e.g., 'ca_14_193.87') into compact label (e.g., 'ca14193_87') ===
                element_label = (
                    line.replace("ca_14_193.87", "ca14193_87")
                        .replace("fe_16_262.98", "fe16262_98")
                        .replace("s_11_188.68", "s11188_68")
                        .replace("si_10_258.37", "si10258_37")
                        .replace("ar_14_194.40", "ar14194_40")  
                        .replace("ar_11_188.81", "ar11188_81")  
                        .replace("s_10_264.23", "s10264_23")   
                        .replace("s_13_256.69", "s13256_69")
                )

                # === Extract datetime from filename (e.g., 'eis_20140202_122934.data.h5') ===
                datetime_str = eis_filename_to_timestamp(Path(filename))

                # === Final filename ===
                fits_filename = f"{datetime_str}_{element_label}.fits"
                fits_path = os.path.join(custom_intensity_dir, fits_filename)
                #if m.data is None:
                #    print(f"ERROR: No data returned for line={line}")
                #elif not np.any(np.isfinite(m.data)):
                #    print(f"ERROR: Data for line={line} is all NaNs or non-finite values")
                #elif np.all(m.data == 0):
                #    print(f"WARNING: Data for line={line} is all zeros")
                #else:
                #    print(f"Data looks valid — attempting to save FITS")

                #    # Save intensity using the same header
                #    Map(I_data, m.meta).save(fits_path, overwrite=True)

                #    # Save the matching error map next to it: *_err.fits
                #    err_fits_path = Path(fits_path).with_name(Path(fits_path).stem + "_err.fits")
                #    save_error_map_like(Map(I_data, m.meta), I_err, err_fits_path)


                print(f"DEBUG: Intensity Stats for {line} -> Min={np.nanmin(I_data)}, Max={np.nanmax(I_data)}, Mean={np.nanmean(I_data)}")
                print(f"DEBUG: Nonzero pixel count for {line}: {np.count_nonzero(np.isfinite(I_data) & (I_data != 0))}")

                if I_data is None:
                    print(f"ERROR: No data returned for line={line}")
                elif not np.any(np.isfinite(I_data)):
                    print(f"ERROR: Data for line={line} is all NaNs or non-finite values")
                elif np.all(I_data == 0):
                    print(f"WARNING: Data for line={line} is all zeros")
                else:
                    print(f"Data looks valid — attempting to save FITS")

                    # Save intensity using the Map header
                    Map(I_data, m.meta).save(out_fits.as_posix(), overwrite=True)

                    # Save the matching error map next to it: *_err.fits
                    save_error_map_like(Map(I_data, m.meta), I_err, err_fits.as_posix())


                print("============================================")
                print(f"Saved intensity map for line={line} in file={filename}")
                print(f"Output location: {custom_intensity_dir}")
                print("============================================")

            except Exception as e:
                print(f"Error generating intensity for line={line} in {filename}: {e}")

#data_dir = (Path.home() / "sunpy" / "data").resolve()


def main():
    parser = argparse.ArgumentParser(description="Generate intensity maps for lines used in composition.")
    parser.add_argument('-c', '--cores', type=int, default=4, help='Number of cores to use.')
    parser.add_argument('--test', action="store_true", help="Run in test mode (only ca_14_193.87 on eis_20140206_234547)")
    args = parser.parse_args()

    line_databases = {
        "sis": ['si_10_258.37', 's_10_264.23', 'SiX_SX'],
        "sar": ['s_11_188.68', 'ar_11_188.81', 'SXI_ArXI'],
        "CaAr": ['ca_14_193.87', 'ar_14_194.40', 'CaXIV_ArXIV'],
        "FeS": ['fe_16_262.98', 's_13_256.69', 'FeXVI_SXIII'],
    }

    if args.test:
        filenames = ["SO_EIS_data/eis_20140202_122934.data.h5"]
    else:
        if not data_dir.exists():
            print(f"ERROR: data dir not found: {data_dir}")
            return
        files = list_eis_data_files(data_dir)
        if not files:
            print(f"ERROR: no EIS *.data.h5 under: {data_dir}")
            return
        filenames = [f.as_posix() for f in files]


    # Process each file
    for filename_full in filenames:

        filename = filename_full.replace(" [processing]", "").replace(" [processed]", "")
        if not filename:
            continue

        print(f"\n==========\nProcessing file: {filename}\n==========")
        try:
            make_intensity_maps_for_file(filename, line_databases, ncpu=args.cores, test_mode=args.test)
        except Exception as e:
            print(f"Error while making intensity maps for {filename}: {e}")

if __name__ == "__main__":
    main()
