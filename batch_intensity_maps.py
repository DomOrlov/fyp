#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
import argparse
import numpy as np
from ashmcmc import ashmcmc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
from sunpy.map import Map
from astropy.io import fits
import re
import multiprocessing

data_dir = Path("/mnt/scratch/data/orlovsd2/sunpy/data").resolve()
fe12_dir = data_dir / "nonaligned_fe12_intensity_maps"
custom_intensity_dir = data_dir / "intensity_map"
test_mode = True
test_target = "2015_10_18__12_49_39"

line_databases = {
    "sis": ['si_10_258.37', 's_10_264.23', 'SiX_SX'],
    "sar": ['s_11_188.68', 'ar_11_188.81', 'SXI_ArXI'],
    "CaAr": ['ca_14_193.87', 'ar_14_194.40', 'CaXIV_ArXIV'],
    "FeS": ['fe_16_262.98', 's_13_256.69', 'FeXVI_SXIII'],
}

# robust timestamp parser
_TS_RE = re.compile(r"eis_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.data\.h5$")

def eis_filename_to_timestamp(file_path: Path) -> str:
    m = _TS_RE.search(file_path.name)
    if not m:
        raise ValueError(f"Unrecognised EIS filename: {file_path.name}")
    Y, M, D, h, m, s = m.groups()
    return f"{Y}_{M}_{D}__{h}_{m}_{s}"

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

def _work(filename):
    # Generate intensity maps for all lines in "line_databases" for the given EIS file.
    a = ashmcmc(str(filename), ncpu=1)
    print(f"ashmcmc outdir => {a.outdir}")
    # Fe XII 195.12 (raw) for this file
    timestamp = eis_filename_to_timestamp(Path(filename))
    fe12_line = "fe_12_195.12"
    fe12_out = fe12_dir / f"eis_{timestamp}_intensity.fits"

    if fe12_out.exists():
        print(f"skipping Fe XII already exists: {fe12_out.name}")
    else:
        print(f"\nGenerating Fe XII {fe12_line} for {filename}")
        m_fe = a.ash.get_intensity(
            fe12_line,
            outdir=fe12_dir,
            refit=False,
            plot=True,
            mcmc=False,
            calib=True,
            calib_year="2014"
        )
        Map(m_fe.data, m_fe.meta).save(fe12_out, overwrite=True)
        print(f"Saved Fe XII: {fe12_out}")

    for ratio_lines in line_databases.values():
        for line in ratio_lines[:2]:
            print(f"\nGenerating intensity map for {line} in {filename}")
            out_fits = custom_intensity_dir / f"{timestamp}_{alias.get(line, line.replace('.', '_'))}.fits"
            err_fits = out_fits.with_name(out_fits.stem + "_err.fits")

            if out_fits.exists() and err_fits.exists():
                print(f"skipping already exists: {out_fits.name} (+err)")
                continue
            try:
                I_data, I_err = a.ash.get_intensity(
                    line,
                    outdir=custom_intensity_dir,
                    refit=(not out_fits.exists() or not err_fits.exists()),
                    plot=True,
                    mcmc=True,
                    calib=True,
                    calib_year="2014"
                )
            except Exception as e:
                print(f"[SKIP LINE] {timestamp} {line}: {e}")
                continue

            m = a.ash.get_intensity(
                line,
                outdir=custom_intensity_dir,
                refit=False,
                plot=False,
                mcmc=False,        
                calib=True,
                calib_year="2014"
            )

            #print(f"Intensity Stats for {line} -> Min={m.data.min()}, Max={m.data.max()}, Mean={m.data.mean()}")
            #print(f"Nonzero pixel count for {line}: {m.data.nonzero()[0].size}")
            # Force-save fits file
            # Convert line (e.g., 'ca_14_193.87') into compact label (e.g., 'ca14193_87') 

            #if m.data is None:
            #    print(f"No data returned for line={line}")
            #elif not np.any(np.isfinite(m.data)):
            #    print(f"Data for line={line} is all NaNs or non-finite values")
            #elif np.all(m.data == 0):
            #    print(f"Data for line={line} is all zeros")
            #else:
            #    print(f"Data looks valid attempting to save fits")

            #    # Save intensity using the same header
            #    Map(I_data, m.meta).save(fits_path, overwrite=True)

            #    # Save the matching error map next to it: *_err.fits
            #    err_fits_path = Path(fits_path).with_name(Path(fits_path).stem + "_err.fits")
            #    save_error_map_like(Map(I_data, m.meta), I_err, err_fits_path)


            print(f"Intensity Stats for {line} -> Min={np.nanmin(I_data)}, Max={np.nanmax(I_data)}, Mean={np.nanmean(I_data)}")
            print(f"Nonzero pixel count for {line}: {np.count_nonzero(np.isfinite(I_data) & (I_data != 0))}")

            if I_data is None:
                print(f"No data returned for line={line}")
            elif not np.any(np.isfinite(I_data)):
                print(f"Data for line={line} is all NaNs or non-finite values")
            elif np.all(I_data == 0):
                print(f"Data for line={line} is all zeros")
            else:
                print(f"Data looks valid attempting to save fits")


                Map(I_data, m.meta).save(out_fits, overwrite=True)
                h = m.meta.copy()
                h["BUNIT"] = str(h.get("BUNIT", "")) + " (error)"
                fits.writeto(err_fits, data=I_err.astype(float), header=fits.Header(h), overwrite=True)

            print(f"Saved intensity map for line={line} in file={filename}")
            print(f"Output location: {custom_intensity_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate intensity maps for lines used in composition.")
    parser.add_argument("-c", "--cores", type=int, default=4, help="Number of cores to use.")
    args = parser.parse_args()
    jobs = sorted(data_dir.rglob("eis_*.data.h5"))
    if test_mode:
        jobs = [f for f in jobs if eis_filename_to_timestamp(f) == test_target]
    if args.cores == 1:
        for fn in jobs:
            _work(fn)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.cores, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_work, jobs, chunksize=1):
                pass