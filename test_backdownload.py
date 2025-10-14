#!/usr/bin/env python3
# -*- coding: utf-8 -*-

ar_line = "AR 12436 2015-10-18 -> 2015-10-30 ~ Beta-Gamma -> Beta-Delta -> Beta-Gamma -> Beta ~ 544"

import re
from datetime import datetime, timedelta
from pathlib import Path
import h5py
from astropy.time import Time
import astropy.units as u
from sunpy.net import Fido, attrs as a
from eispac.net.attrs import FileType
from pfss.functions_data import hmi_daily_download
from iris_get_pfss_utils import get_closest_aia as closest_aia_193
from parfive import Downloader
import glob
import sunpy.map as smap


# -----------------------------------------
# parse AR line and build study allow-list
# -----------------------------------------
m_id  = re.search(r"AR\s+(\d+)", ar_line)
m_rng = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", ar_line)

ar_id    = m_id.group(1)
start_dt = datetime.strptime(m_rng.group(1), "%Y-%m-%d")
end_dt   = datetime.strptime(m_rng.group(2), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)

study_id = [
    654,653,652,648,645,632,624,620,616,613,608,593,592,591,590,589,587,586,583,581,
    569,565,564,561,557,546,544,525,524,523,522,521,520,519,496,468,462,454,437,428,
    427,426,420,405,404,403,362,351,272,263,240,237,230,228,227,195,194,145,110,109,
    108,102,101,98,50,49,39,33,17
]
study_set = set(study_id)

print("====================================")
print("ar id     :", ar_id)
print("date range:", start_dt, "->", end_dt)

t0 = Time(start_dt); t1 = Time(end_dt)

# ---------------------------------------------------
# helper
# ---------------------------------------------------
def read_study_id(hdr_path):
    with h5py.File(hdr_path, "r") as f:
        # ✔ the location in your files
        if "index/study_id" in f:
            v = f["index/study_id"][()]
            if hasattr(v, "shape") and v.shape != ():
                v = v.reshape(()).item()  # handles array([437], dtype=int32)
            return int(v)
        # fallbacks (some datasets use these)
        for p in ("/header/STUDY_ID","/header/study_id",
                  "/eis_header/STUDY_ID","/eis_header/study_id",
                  "/header/PROGRAM_ID","/header/PROG_NUM"):
            if p in f:
                val = f[p][()]
                if hasattr(val, "shape") and val.shape != ():
                    val = val.reshape(()).item()
                return int(val)
    return None

_hmi_mem_cache = {}  # in-memory per-run cache

def hmi_day_cached(day_time):
    day = Time(day_time).strftime("%Y-%m-%d")
    if day in _hmi_mem_cache:
        return _hmi_mem_cache[day]

    # on-disk cache: look in ./data first
    pat = f"./data/hmi.mrdailysynframe_720s.{day.replace('-','')}*.fits"
    files = sorted(glob.glob(pat))
    if files:
        m = smap.Map(files[-1])    # load latest matching file
        _hmi_mem_cache[day] = m
        return m

    # fall back to your existing downloader (does the JSOC request)
    m = hmi_daily_download(Time(day_time))
    _hmi_mem_cache[day] = m
    return m

# ----------------
# 1) EIS HEADERS
# ----------------
print("\nEIS search (Level-1 headers)…")
eis_hdr_res = Fido.search(
    a.Time(t0, t1),
    a.Instrument("EIS"),
    a.Physobs.intensity,
    a.Source("Hinode"),
    a.Provider("NRL"),
    a.Level("1"),
    FileType("HDF5 header")
)
hdr_cnt = len(eis_hdr_res[0]) if len(eis_hdr_res) else 0
print("total headers:", hdr_cnt)

if hdr_cnt == 0:
    print("kept: 0")
    print("====================================")
    raise SystemExit(0)

# fetch headers (small)
hdr_files = Fido.fetch(eis_hdr_res)
print("downloaded headers:", len(hdr_files))

print("example study_id from first header:", read_study_id(hdr_files[0]))

# time column from the header results table
tcol = [c for c in eis_hdr_res[0].colnames if "time" in c.lower()][0]

# ------------------------------
# 2) FILTER BY STUDY_ID FROM HDR
# ------------------------------
kept_idx     = []
kept_times   = []
kept_hdrpath = []

for i, hdr_path in enumerate(hdr_files):
    sid = read_study_id(hdr_path)
    if (sid is not None) and (sid in study_set):
        kept_idx.append(i)
        kept_times.append(Time(str(eis_hdr_res[0][tcol][i])))
        kept_hdrpath.append(hdr_path)

print("kept (by STUDY_ID):", len(kept_idx))
for i in kept_idx[:10]:
    print(" -", Path(hdr_files[i]).name)
if len(kept_idx) > 10:
    print(" - …")

if not kept_times:
    print("No headers matched allowed study IDs; done.")
    print("====================================")
    raise SystemExit(0)

# ----------------------------------------
# 3) DOWNLOAD MATCHING EIS LEVEL-1 *DATA*
# ----------------------------------------
print("\nFetching matching EIS Level-1 DATA…")
kept_data_files = []
for t in kept_times:
    # tight window around header time to get corresponding data file
    r = Fido.search(
        a.Time(t - 1*u.s, t + 1*u.s),
        a.Instrument("EIS"),
        a.Physobs.intensity,
        a.Source("Hinode"),
        a.Provider("NRL"),
        a.Level("1"),
        FileType("HDF5 data"),
    )
    if len(r) and len(r[0]):
        files = Fido.fetch(r[0][0:1])
        if len(files):
            kept_data_files.append(files[0])

print("downloaded data files:", len(kept_data_files))
for p in kept_data_files[:10]:
    print(" -", Path(p).name)
if len(kept_data_files) > 10:
    print(" - …")

# --------------------------------------
# 4) NEAREST AIA 193 FOR EACH KEPT TIME
# --------------------------------------
#dlr_aia = Downloader(max_conn=3, retry=5, progress=True)
dlr_aia = Downloader(max_conn=3, progress=True)
dlr_aia.retry = 5

print("\nnearest AIA 193")
for t in kept_times:
    row = closest_aia_193(t.datetime)  # local lookup
    if row is None:
        w0 = t - 60*u.s
        w1 = t + 60*u.s
        r  = Fido.search(a.Time(w0, w1), a.Instrument.aia, a.Wavelength(193*u.angstrom))
        n  = len(r[0]) if len(r) > 0 else 0
        if n == 0:
            print("EIS", t, "-> AIA:", "none")
            continue
        # pick nearest
        best = min(range(n), key=lambda j: abs(Time(str(r[0]['Start Time'][j])) - t))
        sel = r[0][best:best+1]
        files = Fido.fetch(sel, downloader=dlr_aia)
        if files and len(files):
            print("EIS", t, "-> AIA (fetched):", files[0])
        else:
            print("EIS", t, "-> AIA: fetch FAILED (timeout/no file)")
    else:
        print("EIS", t, "-> AIA (local):", row)


# --------------------------------------
# 5) NEAREST DAILY HMI (TODAY vs YEST)
# --------------------------------------
print("\nnearest HMI (daily: today vs yesterday)")
for t in kept_times:
    hmi_today = hmi_day_cached(t)
    hmi_yest  = hmi_day_cached(t - 1*u.day)

    def time_of(obj):
        # Return astropy Time or None
        if obj is None:
            return None

        # Case A: SunPy Map-like (e.g., HMIMap)
        if hasattr(obj, "meta"):
            # many Maps expose .date directly
            if hasattr(obj, "date") and obj.date is not None:
                return obj.date
            # fall back to common header keys
            for k in ("T_REC", "DATE-OBS", "DATE_OBS", "DATE"):
                if k in obj.meta:
                    try:
                        return Time(obj.meta[k])
                    except Exception:
                        pass
            return None

        # Case B: Table-like
        if hasattr(obj, "colnames"):
            tcols = [c for c in list(obj.colnames) if "time" in c.lower()]
            if tcols:
                try:
                    return Time(str(obj[tcols[0]]))
                except Exception:
                    return None

        return None

    Tt = time_of(hmi_today)
    Ty = time_of(hmi_yest)

    if (Tt is None) and (Ty is None):
        choice = None
    elif (Tt is not None) and (Ty is None):
        choice = hmi_today
    elif (Tt is None) and (Ty is not None):
        choice = hmi_yest
    else:
        choice = hmi_today if abs(Tt - t) <= abs(Ty - t) else hmi_yest

    print("EIS", t, "-> HMI:", "none" if choice is None else choice)


print("====================================")
