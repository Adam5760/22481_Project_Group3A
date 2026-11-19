# FILE: measure.py
# DESCRIPTION: Functions for measurements in images and volumes: distances, line profiles,
#              widths and edge sharpness, and ROI statistics.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Distances and line profiles
# ---------------------------------------------------------------------------
def distance_mm(p1_xy: tuple, p2_xy: tuple, spacing: tuple) -> float:
    """
    From the Lesson 9 Image Analysis Toolbox.

    Distance in mm between two pixel positions in a 2D image.

    Positions are given as (x, y) = (column, row). The spacing must be verified first.
    Suitable for a 2D image with orthogonal pixel axes; not a 3D distance.

    Parameters
    ----------
    p1_xy, p2_xy : tuple
        Pixel positions (column, row).
    spacing : tuple
        (row spacing, column spacing) in mm.

    Returns
    -------
    float
        Distance in mm.

    Example
    -------
    >>> distance_mm((10, 20), (110, 20), spacing=(0.5, 0.5))
    50.0
    """
    if spacing is None:
        raise ValueError("Physical pixel spacing not verified")
    dy, dx = spacing
    # Convert the horizontal and vertical pixel differences to mm separately
    return float(np.hypot((p2_xy[0] - p1_xy[0]) * dx, (p2_xy[1] - p1_xy[1]) * dy))


def line_profile(img: np.ndarray, p1_xy: tuple, p2_xy: tuple, spacing: tuple | None = None,
                 samples: int = 300):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Sample the intensities along a line between two (column, row) positions in a 2D image.

    Uses nearest-neighbour sampling, so repeated pixels and stair-step edges are possible.
    Choose the end points based on the scientific question.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    p1_xy, p2_xy : tuple
        Start and end position (column, row) in pixels.
    spacing : tuple or None
        (row spacing, column spacing) in mm. Default None gives the distance in pixels.
    samples : int
        Number of points along the line. Default is 300.

    Returns
    -------
    dist : np.ndarray
        Distance along the line (mm, or pixels if no spacing is given).
    values : np.ndarray
        Intensities along the line.
    unit : str
        "mm" or "pixels".

    Example
    -------
    >>> dist, values, unit = line_profile(image, (40, 150), (250, 150), spacing=(0.273, 0.273))
    >>> plt.plot(dist, values); plt.xlabel(f"Distance ({unit})")
    """
    x = np.linspace(p1_xy[0], p2_xy[0], samples)                    # Column positions along the line
    y = np.linspace(p1_xy[1], p2_xy[1], samples)                    # Row positions along the line
    # Round the positions to the nearest existing pixel (no interpolation)
    xi = np.clip(np.rint(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(int), 0, img.shape[0] - 1)
    values = img[yi, xi]
    if spacing is None:
        dist = np.linspace(0, np.hypot(x[-1] - x[0], y[-1] - y[0]), samples)
        unit = "pixels"
    else:
        dist = np.linspace(0, distance_mm(p1_xy, p2_xy, spacing), samples)
        unit = "mm"
    return dist, values, unit


def line_profile_3d(vol: np.ndarray, spacing: tuple, p0: tuple, p1: tuple):
    """
    Inspired by line_profile from the Lesson 9 Image Analysis Toolbox.

    Intensity profile along a straight line between two points in a 3D volume.

    Unlike the toolbox version, the line can run in any direction through the volume
    (also across slices), and the values are linearly interpolated between voxels,
    which gives smoother edges than nearest-neighbour sampling.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (axis 0, axis 1, axis 2).
    spacing : tuple
        Voxel spacing in mm (axis 0, axis 1, axis 2).
    p0, p1 : tuple
        Start and end point in voxels (axis 0, axis 1, axis 2).

    Returns
    -------
    dist : np.ndarray
        Distance along the line (mm).
    values : np.ndarray
        Interpolated intensities along the line.

    Example
    -------
    >>> dist, values = line_profile_3d(ct_a3, ct_spacing, (65, 30, 40), (65, 70, 40))
    >>> plt.plot(dist, values); plt.xlabel("Distance (mm)")
    """
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length_mm = np.linalg.norm((p1 - p0) * np.array(spacing))      # Line length in mm
    n = max(int(length_mm / min(spacing) * 2), 2)                  # About 2 samples per smallest voxel
    coords = np.linspace(p0, p1, n).T                              # Sample positions, shape (3, n)
    values = ndimage.map_coordinates(vol, coords, order=1)         # Linear interpolation between voxels
    dist = np.linspace(0, length_mm, n)
    return dist, values


def width_crossing(dist: np.ndarray, values: np.ndarray, level: float, above: bool = True) -> float:
    """
    Width (mm) of the part of a profile above (or below) a level.

    Measured as the distance between the first and last sample that passes the level.
    With the level halfway between the filling and the structure, this is similar to
    the full width at half maximum (FWHM).

    Parameters
    ----------
    dist : np.ndarray
        Distance along the profile (mm).
    values : np.ndarray
        Intensities along the profile.
    level : float
        Intensity level.
    above : bool
        True for structures brighter than the level (e.g. a dense object),
        False for structures darker than the level (e.g. air). Default is True.

    Returns
    -------
    float
        Width in mm (0 if no sample passes the level).

    Example
    -------
    >>> half_level = (filling_level + object_level) / 2
    >>> width_crossing(dist, values, half_level, above=False)
    18.6
    """
    if above:
        passed = values > level                                    # Samples brighter than the level
    else:
        passed = values < level                                    # Samples darker than the level
    d = dist[passed]
    if d.size == 0:
        return 0.0
    return float(d.max() - d.min())                                # First to last passing sample


def edge_width(dist: np.ndarray, values: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """
    10–90 % edge width (mm) of a profile that crosses one edge.

    The profile is normalised between its minimum and maximum. The edge width is the
    distance between the first points where the profile reaches `low` and `high` of the
    edge height. A sharper edge gives a smaller value, so it is a simple measure of
    spatial resolution.

    Parameters
    ----------
    dist : np.ndarray
        Distance along the profile (mm).
    values : np.ndarray
        Intensities along a line that crosses ONE edge.
    low, high : float
        Fractions of the edge height. Default is 0.1 and 0.9.

    Returns
    -------
    float
        Edge width in mm (NaN if the profile is flat).

    Example
    -------
    >>> dist, values = line_profile_3d(ct_a3, ct_spacing, (65, 50, 10), (65, 50, 30))
    >>> edge_width(dist, values)
    1.12
    """
    span = values.max() - values.min()
    if span == 0:                                                  # Flat profile, no edge
        return np.nan
    v = (values - values.min()) / span                             # Normalise to 0-1

    if v[0] > v[-1]:                                               # Make the edge rising
        v, dist = v[::-1], dist.max() - dist[::-1]

    def first_crossing(level):
        # Distance where the profile first reaches the level (interpolated between samples)
        i = int(np.argmax(v >= level))
        if i == 0:
            return dist[0]
        return np.interp(level, [v[i - 1], v[i]], [dist[i - 1], dist[i]])

    return float(abs(first_crossing(high) - first_crossing(low)))

# ---------------------------------------------------------------------------
# Regions of interest (ROIs)
# ---------------------------------------------------------------------------
def roi_values(img: np.ndarray, roi: tuple) -> np.ndarray:
    """
    From the Lesson 9 Image Analysis Toolbox.

    Return the pixel intensities inside a rectangular ROI in a 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    roi : tuple
        (x, y, width, height) in pixels; x is the column, y is the row.
        The rectangle must lie completely inside the image.

    Returns
    -------
    np.ndarray
        1D array of the pixel values in the ROI.

    Example
    -------
    >>> roi_values(image, (300, 200, 20, 20)).mean()
    547.3
    """
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        raise ValueError(f"ROI outside image: {roi}")
    # Array indexing is [row, column], the reverse of (x, y) notation
    return np.asarray(img[y:y + h, x:x + w], float).ravel()


def roi_summary(img: np.ndarray, roi: tuple) -> dict:
    """
    From the Lesson 9 Image Analysis Toolbox.

    Summarise a rectangular 2D ROI with count, mean, median and within-ROI standard deviation.

    The output is in the units of the image. The standard deviation includes ALL
    variation in the ROI (structure, artefacts and noise), so it is not automatically
    a noise estimate.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    roi : tuple
        (x, y, width, height) in pixels.

    Returns
    -------
    dict
        n_pixels, mean, std_within_roi and median.

    Example
    -------
    >>> roi_summary(image, (300, 200, 20, 20))
    {'n_pixels': 400, 'mean': 547.3, 'std_within_roi': 10.2, 'median': 546.0}
    """
    a = roi_values(img, roi)
    return {"n_pixels": len(a), "mean": float(a.mean()),
            "std_within_roi": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            "median": float(np.median(a))}


def roi_stats(vol: np.ndarray, center: tuple, half_size: tuple) -> dict:
    """
    Inspired by roi_summary from the Lesson 9 Image Analysis Toolbox.

    Statistics of a box-shaped ROI in a 3D volume, given by its centre and half-size.

    The ROI goes from centre − half to centre + half along each axis, so it is
    2 × half + 1 voxels wide and always centred on the chosen point. As in the toolbox,
    the standard deviation includes ALL variation in the ROI, so it is only a noise
    estimate if the ROI is homogeneous.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (axis 0, axis 1, axis 2).
    center : tuple of int
        ROI centre in voxels.
    half_size : tuple of int
        Half-size of the ROI in voxels.

    Returns
    -------
    dict
        mean, median, std (within the ROI) and n (number of voxels).

    Example
    -------
    >>> roi_stats(t1_a3, center=(86, 262, 82), half_size=(3, 7, 7))
    {'mean': 547.3, 'median': 546.0, 'std': 10.2, 'n': 1575}
    """
    # The ROI as one slice per axis (clipped at 0, so it never wraps around)
    sl = tuple(slice(max(c - h, 0), c + h + 1) for c, h in zip(center, half_size))
    values = vol[sl].astype(float).ravel()                         # All voxel values in the ROI

    return {"mean":   float(values.mean()),
            "median": float(np.median(values)),
            "std":    float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "n":      int(values.size)}


def roi_stability(vol: np.ndarray, center: tuple, half_size: tuple, shift: int = 2) -> tuple:
    """
    Range of the ROI mean when the ROI is moved and resized.

    Answers the toolbox's question "would moving an ROI affect the result?". The ROI is
    moved −shift, 0 and +shift voxels along each axis (27 positions), and its half-size
    is changed by −1 and +1 voxel.

    Parameters
    ----------
    vol : np.ndarray
        3D volume.
    center : tuple of int
        ROI centre in voxels.
    half_size : tuple of int
        Half-size of the ROI in voxels.
    shift : int
        Number of voxels to move the ROI. Default is 2.

    Returns
    -------
    tuple
        (lowest mean, highest mean) over all tested positions and sizes.

    Example
    -------
    >>> roi_stability(t1_a3, center=(86, 262, 82), half_size=(3, 7, 7))
    (541.8, 552.6)
    """
    means = []

    # Move the ROI along each axis
    for s0 in (-shift, 0, shift):
        for s1 in (-shift, 0, shift):
            for s2 in (-shift, 0, shift):
                moved = (center[0] + s0, center[1] + s1, center[2] + s2)
                means.append(roi_stats(vol, moved, half_size)["mean"])

    # Make the ROI one voxel smaller and larger
    for d in (-1, 1):
        resized = tuple(max(h + d, 0) for h in half_size)
        means.append(roi_stats(vol, center, resized)["mean"])

    return float(min(means)), float(max(means))