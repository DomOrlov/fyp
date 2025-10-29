#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta
from pathlib import Path
import h5py
from astropy.time import Time
import astropy.units as u
from sunpy.net import Fido, attrs as a
from eispac.net.attrs import FileType
from pfss.functions_data import PrepHMIdaily, hmi_daily_download
from iris_get_pfss_utils import get_closest_aia as closest_aia_193
from parfive import Downloader
import glob
import sunpy.map as smap
from astropy.io import fits

import os
import traceback  # for stack traces in excepts

error_log = []
LOG_FILE = Path("errorlog.txt")

def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} | {msg}"
    print(msg)                  # keep console output as-is
    error_log.append(line)      # keep in-memory copy

    # also persist to disk
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        # last-resort: show why logging failed
        print(f"!! failed to write to {LOG_FILE}: {e}")

#ar_line = "AR 12436 2015-10-18 -> 2015-10-30 ~ Beta-Gamma -> Beta-Delta -> Beta-Gamma -> Beta ~ 544"

ar_catalogue = [
"AR 11967 2014-01-28 -> 2014-02-10 ~ Beta -> Beta-Gamma -> Beta-Gamma-Delta ~ 437, 403",
"AR 12434 2015-10-14 -> 2015-10-26 ~ (Till 25) Beta -> Beta-Gamma -> Beta-Gamma-Delta -> Beta ~ 544",
"AR 12436 2015-10-18 -> 2015-10-30 ~ Beta-Gamma -> Beta-Delta -> Beta-Gamma -> Beta ~ 544",
"AR 11504 2012-06-10 -> 2012-06-22 ~ Beta -> Beta-Gamma-Delta Beta-Gamma -> Beta ~ 404, 420",
"AR 12665 2017-07-05 -> 2017-07-17 ~ (06) Alpha -> Beta -> Beta-Gamma -> Beta ~ 544",
"AR 12759 2020-04-01 -> 2020-04-06 ~ (Till 05) Alpha ~ 569, 544",
"AR 12351 2015-05-20 -> 2015-05-28 ~ Beta -> Dropped -> Alpha -> Dropped (For the rest of the days) -> ~ 437, 414",
"AR 12107 2014-06-30 -> 2014-07-13 ~ Beta -> Beta-Gamma -> Beta-Delta -> Beta -> Beta-Gamma -> Beta -> Alpha ~ 404, 420",
"AR 12132 2014-08-05 -> 2014-08-07 ~ Beta-Gamma -> Beta ~ 404, 420",
"AR 12519 2016-03-12 -> 2016-03-17 ~ Beta -> Alpha -> Beta ~ 404, 520",
"AR 12564 2016-07-09 -> 2016-07-21 ~ Alpha -> Beta -> Beta-Gamma -> Beta -> Alpha -> Beta -> Alpha -> Dropped ~ 404",
"AR 12546 2016-05-20 -> 2016-05-27 ~ Beta -> Alpha ~ 404, 520, 403",
"AR 12289 2015-02-21 -> 2015-02-27 ~ Alpha -> Dropped -> Beta -> Alpha ~ 520, 523 (two)",
"AR 12297 2015-03-07 -> 2015-03-15 ~ Alpha -> Beta-Gamma ->  Beta-Gamma-Delta -> Beta-Delta -> Beta-Gamma-Delta -> Beta-Gamma ~ 520 (eight)",
"AR 12387 2015-07-19 -> 2015-07-27 ~ Beta -> Dropped ~ 520, 404, 522",
"AR 12401 2015-08-16 -> 2015-08-25 ~ Beta -> Alpha -> Dropped ~ 520",
"AR 12429 2015-10-08 -> 2015-10-17 ~ Beta -> Dropped(10th) ~ 520",
"AR 11944 2014-01-02 -> 2014-01-15 ~ Beta -> Beta-Gamma -> Beta-Gamma-Delta -> Beta-Gamma -> Beta ~ 403, 437",
"AR 12381 2015-07-05 -> 2015-07-16 ~ Beta -> Beta-Gamma -> Beta  ~ 522, 404",
"AR 12470 2015-12-19 -> 2015-12-26 ~ Beta -> Alpha ~ 522 (three), 404 (three), 520 (four), 437 (one)",
"AR 12687 2017-11-20 -> 2017-11-25 ~ Dropped ~ 569",
"AR 12685 2017-10-25 -> 2017-11-01 ~ Alpha -> Dropped ~ 569",
"AR 12680 2017-09-14 -> 2017-09-21 ~ Alpha ~ 569",
"AR 12602 2016-10-15 -> 2016-10-21 ~ Beta ~ 544",
"AR 12532 2016-04-21 -> 2016-04-26 ~ Beta -> Alpha -> Dropped -> Beta ~ 522",
"AR 12592 2016-09-17 -> 2016-09-21 ~ Beta -> Dropped ~ 522",
"AR 12672 2017-08-26 -> 2017-08-30 ~ Beta-Gamma -> Beta ~ 521",
"AR 12376 2015-06-30 -> 2015-07-07 ~ (01) Beta-Gamma -> Beta ->  ~ 521",
"AR 12335 2015-05-02 -> 2015-05-06 ~ Beta -> Beta-Gamma ~ 521",
"AR 12524 2016-03-17 -> 2016-03-28 ~ Alpha -> Beta -> Alpha ~ 520",
"AR (No Class) 2012-08-06 -> 2012-08-13 ~ ~ 454",
"AR 11548 2012-08-25 -> 2012-08-28 ~ Beta -> Dropped ~ 454",
"AR 11575 2012-09-25 -> 2012-09-28 ~ Beta-Gamma -> Alpha ~ 454",
"AR 11593 2012-10-18 -> 2012-10-21 ~ Alpha -> Beta ~ 454",
"AR 12362 2015-06-09 -> 2015-06-15 ~ Beta -> Alpha ~ 454",
"AR 12553 2016-06-14 -> 2016-06-18 ~ Alpha ~ 454",
"AR 12574 2016-08-10 -> 2016-08-16 ~ Beta ~ 454",
"AR 11176 2011-03-24 -> 2011-04-02 ~ Beta-Gamma -> Beta ~ 437",
"AR 11204 2011-05-04 -> 2011-05-09 ~ Beta ~ 437",
"AR 11330 2011-10-26 -> 2011-10-31 ~ Beta-Gamma ~ 437",
"AR 11633 2012-12-16 -> 2012-12-24 ~ Beta -> Beta-Gamma -> Beta -> Beta-Gamma ~ 437",
"AR 11490 2012-05-29 -> 2012-06-01 ~ Beta ~ 428",
"AR 11799/800 2013-07-23 -> 2013-07-27 ~ Alpha/Beta -> Alpha/Beta-Gamma -> Dropped/Beta-Gamma ~ 428",
"AR 12063 2014-05-17 -> 2014-05-22 ~ Beta -> Alpha -> Dropped ~ 428",
"AR 12135 2014-08-09 -> 2014-08-14 ~ Beta-Gamma -> Beta ~ 428",
"AR 11512 2012-06-25 -> 2012-06-28 ~ (26) Beta -> Beta-Gamma ~ 420",
"AR 11584 2012-09-29 -> 2012-10-05 ~ (01) Beta -> Alpha ~ 420",
"AR 11739 2013-05-07 -> 2013-05-11 ~ Beta-Gamma -> Beta ~ 420",
"AR 12087 2014-06-15 -> 2014-06-20 ~ Beta-Gamma -> Beta -> Beta-Gamma -> Beta ~ 420",
"AR 12104/07 2014-07-01 -> 2014-07-07 ~ Beta-Gamma-Delta/Beta-Gamma -> Beta-Gamma-Delta/Beta -> Beta-Gamma/Beta ->  Beta-Gamma/Beta-Gamma ~ 420 404",
"AR 12282 2015-02-14 -> 2015-02-20 ~ Beta-Gamma -> Beta -> Alpha ~ 420",
"AR 12333 2015-04-25 -> 2015-05-02 ~ Alpha -> Beta -> Dropped~ 420",
"AR 12371 2015-06-20 -> 2015-06-29 ~ Beta-Gamma-Delta -> Beta-Gamma -> Beta ~ 420",
"AR 12539 2016-05-01 -> 2016-05-06 ~ Beta -> Alpha ~ 420",
"AR 11785 2013-07-08 -> 2013-07-13 ~ Beta-Gamma-Delta -> Beta-Gamma ~ 404",
"AR 11838 2013-09-07 -> 2013-09-11 ~ Alpha ~ 404",
"AR 12573 2016-08-12 -> 2016-08-16 ~ [(08-09) Beta] ~ 404",
]

# ---------------------------------------------------
# helper
# ---------------------------------------------------
def read_study_id(hdr_path):
    with h5py.File(hdr_path, "r") as f:
        # the location in your files
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
    """
    Return a prepped HMI daily synoptic Map closest to `day_time`,
    searching multiple on-disk roots before falling back to JSOC.
    Always run the file through PrepHMIdaily to fix headers (avoids 'sin(deg)' unit error).
    """
    day = Time(day_time).strftime("%Y-%m-%d")
    if day in _hmi_mem_cache:
        return _hmi_mem_cache[day]

    # Where to look
    roots = [
        Path("./hmi_data"),
        Path.home() / "sunpy" / "data",
        Path.home() / "fyp" / "data",
        Path.home() / "intra" / "pfss" / "data",
    ]
    from os import getenv
    sdd = getenv("SUNPY_DOWNLOAD_DIR")
    if sdd:
        roots.append(Path(sdd))

    # What to look for (cover compact + dotted dates; allow extra suffixes like .data.fits)
    y, m, d = day.split("-")
    pats_rel = [
        f"hmi.mrdailysynframe_720s.{y}{m}{d}_*.fits",   # e.g. ...20151022_120000_TAI.data.fits
        f"hmi.mrdailysynframe_720s.{y}{m}{d}*.fits",    # compact without underscore-time (belt & braces)
        f"hmi.mrdailysynframe_720s.{y}.{m}.{d}*.fits",  # dotted date variant
        f"*mrdailysynframe*{y}{m}{d}*.fits",            # safety nets
        f"*mrdailysynframe*{y}.{m}.{d}*.fits",
    ]

    hits = []
    for root in roots:
        for pat in pats_rel:
            hits.extend(glob.glob(str(root / pat)))

    hits = sorted(set(hits))
    if hits:
        chosen = str(Path(hits[-1]))  # latest if multiple
        print(f"[HMI cache] using disk: {chosen}")
        # IMPORTANT: run through your header-fixing routine
        m = PrepHMIdaily(chosen)
        _hmi_mem_cache[day] = m
        return m

    # Fallback: download (this already calls PrepHMIdaily inside hmi_daily_download)
    print("[HMI cache] JSOC download")
    m = hmi_daily_download(Time(day_time))
    _hmi_mem_cache[day] = m
    return m

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

dlr_aia = Downloader(max_conn=3, progress=True)
dlr_aia.retry = 5

from parfive import Downloader
eis_dlr = Downloader(max_conn=1, max_splits=1, progress=True)
eis_dlr.retry = 5


# ---------------------------------------------------
# main loop over ARs
# ---------------------------------------------------

#for ar_line in ar_catalogue:
#	# -----------------------------------------
#	# parse AR line and build study allow-list
#	# -----------------------------------------
#	m_id  = re.search(r"AR\s+(\d+)", ar_line)
#	#if not m_id:
#	#	print(f"SKIP: no numeric AR id in line: {ar_line}")
#	#	continue
#	#m_rng = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", ar_line)


#	#ar_id    = m_id.group(1)
#	#start_dt = datetime.strptime(m_rng.group(1), "%Y-%m-%d")
#	m_id = re.search(r"AR\s+(\d+)", ar_line)
#	ar_id = m_id.group(1) if m_id else "(No Class)"

#	m_rng = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", ar_line)
#	if not m_rng:
#		print(f"SKIP: no date range in line: {ar_line}")
#		continue

#	start_dt = datetime.strptime(m_rng.group(1), "%Y-%m-%d")

#	end_dt   = datetime.strptime(m_rng.group(2), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)

for ar_line in ar_catalogue:
	# -----------------------------------------
	# parse AR line and build study allow-list
	# -----------------------------------------
	m_id = re.search(r"AR\s+(\d+)", ar_line)
	ar_id = m_id.group(1) if m_id else "(No Class)"

	m_rng = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", ar_line)
	if not m_rng:
		print(f"SKIP: no date range in line: {ar_line}")
		continue

	start_dt = datetime.strptime(m_rng.group(1), "%Y-%m-%d")
	end_dt = datetime.strptime(m_rng.group(2), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)


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
		log(f"[{ar_id}] No EIS headers found for {t0} -> {t1}")
		print("kept: 0")
		print("====================================")
		# raise SystemExit(0)
		continue


	# fetch headers (small)
	hdr_files = Fido.fetch(eis_hdr_res, downloader=eis_dlr)
	print("downloaded headers:", len(hdr_files))

	#print("example study_id from first header:", read_study_id(hdr_files[0]))
	if hdr_files:
		print("example study_id from first header:", read_study_id(hdr_files[0]))
	else:
		log(f"[{ar_id}] Fido.fetch returned no header files (network/cache issue)")



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
		#raise SystemExit(0)
		continue

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
			files = Fido.fetch(r[0][0:1], downloader=eis_dlr)
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

	print("\nnearest AIA 193")
	for t in kept_times:
		row = closest_aia_193(t.datetime)  # local lookup
		if row is None:
			w0 = t - 120*u.s
			w1 = t + 120*u.s
			r  = Fido.search(a.Time(w0, w1), a.Instrument.aia, a.Wavelength(193*u.angstrom))
			n  = len(r[0]) if len(r) > 0 else 0
			if n == 0:
				log(f"[{ar_id}] No AIA 193 within ±120 s for {t}")
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
				log(f"[{ar_id}] AIA fetch failed (timeout/no file) for {t}")

		else:
			print("EIS", t, "-> AIA (local):", row)


	# --------------------------------------
	# 5) NEAREST DAILY HMI (TODAY vs YEST)
	# --------------------------------------
	print("\nnearest HMI (daily: today vs yesterday)")
	for t in kept_times:
		#hmi_today = hmi_daily_download(t)
		#hmi_yest  = hmi_daily_download(t - 1*u.day)
		hmi_today = hmi_day_cached(t)
		hmi_yest  = hmi_day_cached(t - 1*u.day)

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

		if choice is None:
			log(f"[{ar_id}] No HMI daily map (today/yesterday) suitable for {t}")
		print("EIS", t, "-> HMI:", "none" if choice is None else choice)


	print("====================================")
