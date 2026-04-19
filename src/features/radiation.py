"""Extraterrestrial radiation and daylength (FAO-56 eq. 21 and 34).

These are geometric quantities of the sun; they need only latitude and doy.
Useful as lightweight proxies for solar-radiation features when the sensor
stack lacks a pyranometer.
"""
from __future__ import annotations

import numpy as np

GSC_MJ_M2_MIN = 0.0820


def extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """FAO-56 eq. 21. Returns Ra in MJ/m^2/day."""
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    cos_ws = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    ra = (24.0 * 60.0 / np.pi) * GSC_MJ_M2_MIN * dr * (
        ws * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    return ra


def daylength_hours(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """FAO-56 eq. 34. Returns daylight hours."""
    phi = np.deg2rad(lat_deg)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    cos_ws = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    return (24.0 / np.pi) * ws
