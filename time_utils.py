from astropy.time import Time

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