#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# hard-coded AR line (exact format you prefer)
ar_line = "AR 12436 2015-10-18 -> 2015-10-30 ~ Beta-Gamma -> Beta-Delta -> Beta-Gamma -> Beta ~ 544"

import re
from datetime import datetime, timedelta
from astropy.time import Time
import astropy.units as u
from sunpy.net import Fido, attrs as a

m_id  = re.search(r"AR\s+(\d+)", ar_line)
m_rng = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", ar_line)

ar_id     = m_id.group(1)
start_dt  = datetime.strptime(m_rng.group(1), "%Y-%m-%d")                                 # 00:00:00
end_dt    = datetime.strptime(m_rng.group(2), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)  # 23:59:59

print("====================================")
print("ar id     :", ar_id)
print("date range:", start_dt, "->", end_dt)

t0 = Time(start_dt).isot
t1 = Time(end_dt).isot

# AIA 193 Å
print("\n[AIA 193 Å] searching online…")
aia_res = Fido.search(a.Time(t0, t1), a.Instrument.aia, a.Wavelength(193*u.angstrom))
aia_cnt = len(aia_res[0]) if len(aia_res) > 0 else 0
print("  found:", aia_cnt, "record(s)")
if aia_cnt:
    # show a few examples
    for i in range(min(3, aia_cnt)):
        print("   -", aia_res[0][i])
    if aia_cnt > 3:
        print("   - …")
        print("   -", aia_res[0][-1])

# HMI LOS (PFSS uses LOS)
print("\n[HMI LOS] searching online…")
hmi_res = Fido.search(a.Time(t0, t1), a.Instrument.hmi, a.Physobs.los_magnetic_field)
hmi_cnt = len(hmi_res[0]) if len(hmi_res) > 0 else 0
print("  found:", hmi_cnt, "record(s)")
if hmi_cnt:
    for i in range(min(3, hmi_cnt)):
        print("   -", hmi_res[0][i])
    if hmi_cnt > 3:
        print("   - …")
        print("   -", hmi_res[0][-1])

# Hinode/EIS (to confirm rasters exist in the window)
print("\n[Hinode/EIS] searching online…")
eis_res = Fido.search(a.Time(t0, t1), a.Instrument.eis)
eis_cnt = len(eis_res[0]) if len(eis_res) > 0 else 0
print("  found:", eis_cnt, "record(s)")
if eis_cnt:
    for i in range(min(3, eis_cnt)):
        print("   -", eis_res[0][i])
    if eis_cnt > 3:
        print("   - …")
        print("   -", eis_res[0][-1])
