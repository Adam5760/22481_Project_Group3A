# FILE: measure.py
# DESCRIPTION: Measurement helpers for ROIs and line profiles in CT and MRI volumes.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from scipy import ndimage
import numpy as np

# ---------------------------------------------------------------------------
# ROIs
# ---------------------------------------------------------------------------
def roi_stats(vol: np.ndarray, center: tuple, half_size: tuple) -> dict:
    """
    Statistics of a box-shaped ROI.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (z, y, x).
    center : tuple of int
        ROI centre in voxels (z, y, x).
    half_size : tuple of int
        Half-size of the ROI in voxels (z, y, x). The ROI is 2*half+1 voxels wide.

    Returns
    -------
    dict
        Mean, median, within-ROI standard deviation and number of voxels.
        The standard deviation includes ALL variation in the ROI (noise, structure,
        artefacts), so it is only a noise estimate if the ROI is homogeneous.
    """
    sl = tuple(slice(max(c - h, 0), c + h + 1) for c, h in zip(center, half_size))
    values = vol[sl].astype(float).ravel()                              # All voxel values in the ROI

    return {"mean":   float(values.mean()),
            "median": float(np.median(values)),
            "std":    float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "n":      int(values.size)}


def roi_stability(vol: np.ndarray, center: tuple, half_size: tuple, shift: int = 2) -> tuple:
    """
    Range of the ROI mean when the ROI is moved and resized.

    The ROI is moved -shift, 0 and +shift voxels in each direction (27 positions),
    and its half-size is changed by -1 and +1 voxel.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (z, y, x).
    center : tuple of int
        ROI centre in voxels (z, y, x).
    half_size : tuple of int
        Half-size of the ROI in voxels (z, y, x).
    shift : int
        Number of voxels to move the ROI (default 2).

    Returns
    -------
    tuple
        (min mean, max mean) over all tested positions and sizes.
    """
    means = []
    for sz in (-shift, 0, shift):                                       # Move the ROI
        for sy in (-shift, 0, shift):
            for sx in (-shift, 0, shift):
                c = (center[0] + sz, center[1] + sy, center[2] + sx)
                means.append(roi_stats(vol, c, half_size)["mean"])

    for d in (-1, 1):                                                   # Resize the ROI
        h = tuple(max(x + d, 0) for x in half_size)
        means.append(roi_stats(vol, center, h)["mean"])

    return float(min(means)), float(max(means))

# ---------------------------------------------------------------------------
# Line profiles
# ---------------------------------------------------------------------------
def line_profile(vol: np.ndarray, spacing: tuple, p0: tuple, p1: tuple) -> tuple:
    """
    Intensity profile along a straight line between two points.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (z, y, x).
    spacing : tuple
        Voxel spacing in mm (dz, dy, dx).
    p0, p1 : tuple
        Start and end point in voxels (z, y, x).

    Returns
    -------
    dist : np.ndarray
        Distance along the line (mm).
    values : np.ndarray
        Intensities along the line (linear interpolation between voxels).
    """
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length_mm = np.linalg.norm((p1 - p0) * np.array(spacing))          # Line length in mm
    n = max(int(length_mm / min(spacing) * 2), 2)                      # ~2 samples per smallest voxel
    coords = np.linspace(p0, p1, n).T                                  # Shape (3, n)
    values = ndimage.map_coordinates(vol, coords, order=1)             # Linear interpolation
    dist = np.linspace(0, length_mm, n)
    return dist, values


def width_crossing(dist: np.ndarray, values: np.ndarray, level: float, above: bool = True) -> float:
    """
    Width (mm) of the part of a profile above (or below) a level.

    Measured as the distance between the first and last sample that passes the level.

    Parameters
    ----------
    dist : np.ndarray
        Distance along the profile (mm).
    values : np.ndarray
        Intensity profile along the line.
    level : float
        Intensity level.
    above : bool
        True for structures brighter than the level (e.g. a dense object),
        False for structures darker than the level (e.g. air).

    Returns
    -------
    float
        Width in mm (0 if no sample passes the level).
    """
    passed = values > level if above else values < level               # Samples that pass the level
    d = dist[passed]
    return float(d.max() - d.min()) if d.size else 0.0


def edge_width(dist: np.ndarray, values: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """
    10-90 % edge width (mm) of a profile that crosses one edge.

    The profile is normalised between its minimum and maximum. The edge width is the
    distance between the first points where the profile reaches `low` and `high` of
    the edge height. A sharper edge gives a smaller value, so it is a simple measure
    of spatial resolution.

    Parameters
    ----------
    dist : np.ndarray
        Distance along the profile (mm).
    values : np.ndarray
        Intensity profile along a line that crosses ONE edge.
    low : float
        Lower fraction of the edge height (default 0.1).
    high : float
        Upper fraction of the edge height (default 0.9).

    Returns
    -------
    float
        Edge width in mm (NaN if the profile is flat).
    """
    span = values.max() - values.min()
    if span == 0:                                                      # Flat profile, no edge
        return np.nan
    v = (values - values.min()) / span                                 # Normalise to 0-1

    if v[0] > v[-1]:                                                   # Make the edge rising
        v, dist = v[::-1], dist.max() - dist[::-1]

    def first_crossing(level):
        # Distance where the profile first reaches the level (interpolated between samples)
        i = int(np.argmax(v >= level))
        if i == 0:
            return dist[0]
        return np.interp(level, [v[i - 1], v[i]], [dist[i - 1], dist[i]])

    return float(abs(first_crossing(high) - first_crossing(low)))