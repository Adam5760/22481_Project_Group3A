# FILE: data_io.py
# DESCRIPTION: Functions for loading the phantom image data and saving results.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import skimage.color as color

# Project folders, found from this file's location (project/src/data_io.py)
PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
IMAGES  = PROJECT / "images"

# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def get_pixel_spacing(ds: pydicom.dataset.Dataset):
    """
    Inspired by pixel_spacing_mm from the Lesson 9 Image Analysis Toolbox.

    Get the pixel spacing in mm from a DICOM header (CT, MRI, X-ray or ultrasound).

    The spacing is stored in different places depending on the modality:
    PixelSpacing (CT, MRI), ImagerPixelSpacing (X-ray) or
    SequenceOfUltrasoundRegions (ultrasound, given in cm).

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM header.

    Returns
    -------
    tuple or None
        Pixel spacing in mm (row spacing, column spacing), or None if not found.
    """
    # --- Check inputs ---
    if not isinstance(ds, pydicom.dataset.Dataset):
        raise TypeError(f"Input must be a pydicom Dataset, but got {type(ds)}.")

    # 1. CT and MRI: PixelSpacing
    if "PixelSpacing" in ds:
        dy, dx = ds.PixelSpacing

    # 2. X-ray: ImagerPixelSpacing (pixel size at the detector)
    elif "ImagerPixelSpacing" in ds:
        dy, dx = ds.ImagerPixelSpacing

    # 3. Ultrasound: SequenceOfUltrasoundRegions, given in cm
    elif "SequenceOfUltrasoundRegions" in ds:
        region = ds.SequenceOfUltrasoundRegions[0]
        dy = float(region.PhysicalDeltaY) * 10              # cm -> mm
        dx = float(region.PhysicalDeltaX) * 10              # cm -> mm

    # No spacing information found
    else:
        return None

    # 4. Return as floats (row spacing, column spacing)
    return float(dy), float(dx)


def _slice_position(ds: pydicom.dataset.Dataset) -> float:
    """
    Position of a slice along the scan direction, in mm.

    ImagePositionPatient (the corner of the slice in 3D) is projected onto the
    slice normal, which is the cross product of the row and column directions in
    ImageOrientationPatient. This works for any scan orientation.

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM header of one slice.

    Returns
    -------
    float
        Position of the slice along the scan direction (mm).
    """
    orientation = np.array(ds.ImageOrientationPatient, dtype=float)    # Row and column directions (6 numbers)
    row_dir, col_dir = orientation[:3], orientation[3:]                # Split into the two direction vectors
    normal = np.cross(row_dir, col_dir)                                # Normal of the slice plane
    position = np.array(ds.ImagePositionPatient, dtype=float)          # Position of the slice corner in 3D

    return float(np.dot(position, normal))                             # Position along the normal


def slice_spacing_mm(slices: list) -> float:
    """
    Inspired by load_dicom from the Lesson 9 Image Analysis Toolbox.

    Distance between neighbouring slice centres in mm, from the slice positions.

    SliceThickness is not necessarily the distance between slices: slices can
    overlap or have gaps. The spacing is therefore calculated from the positions.

    Parameters
    ----------
    slices : list of pydicom Dataset
        DICOM headers of the slices, sorted by position (as returned by load_dicom).

    Returns
    -------
    float
        Median distance between neighbouring slices (mm).
    """
    # --- Check inputs ---
    if len(slices) < 2:
        raise ValueError("At least two slices are needed to calculate the slice spacing.")

    positions = [_slice_position(ds) for ds in slices]     # Position of each slice along the scan direction
    differences = np.diff(positions)                        # Distance between neighbouring slices
    return float(np.median(np.abs(differences)))            # Median, robust to a single odd gap

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_dicom(path: str | Path):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Read one DICOM image file, and fail clearly if it has no pixel data.

    Parameters
    ----------
    path : str or Path
        DICOM file.

    Returns
    -------
    pydicom Dataset
        The DICOM file, including pixel data.

    Example
    -------
    >>> ds = _read_dicom("data/raw/XRay/image.dcm")
    >>> ds.Modality
    'DX'
    """
    ds = pydicom.dcmread(str(path))                         # Read header and pixel data
    if not hasattr(ds, "PixelData"):                        # Some DICOM files (e.g. reports) have no image
        raise ValueError(f"No image pixel data in {path}")
    return ds


def _pixels(ds: pydicom.dataset.Dataset):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Return the pixel values as floats, with the rescale applied.

    DICOM can store values with a linear conversion: value = stored * slope + intercept.
    For CT this gives Hounsfield units; for other modalities the values are not
    necessarily calibrated.

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM file with pixel data.

    Returns
    -------
    np.ndarray
        Pixel values (float).

    Example
    -------
    >>> image = _pixels(ds)
    >>> image.min(), image.max()
    (-1024.0, 3071.0)
    """
    a = np.asarray(ds.pixel_array).astype(float)            # Stored values as floats
    slope = float(getattr(ds, "RescaleSlope", 1))           # 1 if not given
    intercept = float(getattr(ds, "RescaleIntercept", 0))   # 0 if not given
    return a * slope + intercept                            # Rescaled values


def load_dicom(path: str | Path, series_uid: str | None = None):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Load a DICOM file or a single series from a directory.

    Assumptions: single-frame grayscale images with matching dimensions.
    Check orientation, slice spacing, pixel spacing and units before measuring.

    Parameters
    ----------
    path : str or Path
        A single image file or directory containing DICOM files.
    series_uid : str or None
        Select a specific series when the directory contains several.
        Default is None, which loads the series if there is only one
        (an error is raised if there are several).

    Returns
    -------
    image_or_volume : ndarray
        2D for one file; 3D (slice, row, column) for a directory.
    first_dataset : pydicom Dataset
        Header of the first image in the selected series.
    datasets : list
        DICOM datasets in the same order as the returned slices.

    Example
    -------
    >>> volume, first_ds, datasets = load_dicom(Path("data/raw/CT/Helical"))
    >>> print(volume.shape, first_ds.Rows, first_ds.Columns)
    (522, 512, 512) 512 512
    """
    path = Path(path)

    # --- A single file: return the 2D image ---
    if path.is_file():
        ds = _read_dicom(path)
        return _pixels(ds), ds, [ds]
    if not path.is_dir():
        raise FileNotFoundError(path)

    # --- Group the files by series (headers only, so unrelated series are not mixed) ---
    groups = {}
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.name.startswith("."):       # Skip folders and hidden files (e.g. .DS_Store)
            continue
        try:
            h = pydicom.dcmread(str(p), stop_before_pixels=True)
            if not hasattr(h, "Rows"):                      # Not an image
                continue
            uid = str(getattr(h, "SeriesInstanceUID", "unknown"))
            groups.setdefault(uid, []).append(p)
        except Exception:                                   # Not a DICOM file
            continue
    if not groups:
        raise ValueError("No readable DICOM image files found")

    # --- Choose the series ---
    print("Available series:")
    for uid, paths in groups.items():
        print(f"  {uid} : {len(paths)} files")
    if series_uid is None and len(groups) > 1:
        raise ValueError("Multiple series: set SERIES_UID to one listed above")
    uid = series_uid or next(iter(groups))
    if uid not in groups:
        raise ValueError("SERIES_UID not found")

    # --- Read the selected series and sort the slices by physical position ---
    dsets = [_read_dicom(p) for p in groups[uid]]

    def sort_key(ds):
        """Sort slices by physical position if available; otherwise instance number."""
        try:
            orient = np.asarray(ds.ImageOrientationPatient, float)
            normal = np.cross(orient[:3], orient[3:])                    # Slice normal
            return float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal))
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0))

    dsets.sort(key=sort_key)

    # --- Stack the slices into one 3D array ---
    arrs = [_pixels(ds) for ds in dsets]
    if len({a.shape for a in arrs}) != 1:                    # All slices must have the same size
        raise ValueError("Different image shapes within series")
    return np.stack(arrs, axis=0), dsets[0], dsets


def pixel_spacing_mm(ds: pydicom.dataset.Dataset):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Return the pixel spacing (row spacing, column spacing) in mm/pixel.

    Only reads PixelSpacing (CT, MRI). For X-ray or ultrasound, use
    get_pixel_spacing instead.

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM header.

    Returns
    -------
    tuple or None
        (row spacing, column spacing) in mm, or None if PixelSpacing is absent.

    Example
    -------
    >>> pixel_spacing_mm(first_ds)
    (0.2734, 0.2734)
    """
    if hasattr(ds, "PixelSpacing"):
        return tuple(float(v) for v in ds.PixelSpacing)     # (row, column), NOT (x, y)
    return None


def show_metadata(ds: pydicom.dataset.Dataset, arr: np.ndarray):
    """
    From the Lesson 9 Image Analysis Toolbox.

    Print the image dimensions and the metadata needed to judge whether
    measurements are valid (geometry, calibration and orientation).

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM header.
    arr : np.ndarray
        The loaded image or volume.

    Example
    -------
    >>> show_metadata(first_ds, volume)
    Array shape: (120, 512, 512)
    Modality: MR
    ...
    """
    print("Array shape:", arr.shape)
    for field in ["Modality", "SeriesDescription", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
                  "PhotometricInterpretation", "RescaleSlope", "RescaleIntercept"]:
        print(f"{field}: {getattr(ds, field, 'not provided')}")
    print("WARNING: confirm image geometry, pixel calibration, and orientation before physical measurements.")

# Notes from the toolbox:
# PixelSpacing is (row, column) mm/pixel, NOT (x, y).
# Slice thickness is not necessarily the distance between adjacent slice centres.
# Do not combine different series without checking geometry and scaling.
# This basic loader is not designed for colour ultrasound or multiframe data.


def load_series(folder: str | Path, series_uid: str | None = None):
    """
    Inspired by load_dicom from the Lesson 9 Image Analysis Toolbox.

    Load a CT or MRI series as a 3D array, together with its voxel spacing.

    The slices are loaded and sorted with load_dicom; this function adds the voxel
    spacing in mm, so every volume comes with the spacing needed for measurements.

    Parameters
    ----------
    folder : str or Path
        Folder containing the DICOM slices of one series.
    series_uid : str or None
        Series to load if the folder contains several (see the list printed by load_dicom).
        Default is None, which loads the only series in the folder.

    Returns
    -------
    volume : np.ndarray
        3D array (slices, rows, cols) as float32. CT values are in Hounsfield units.
    spacing : tuple
        Voxel spacing in mm (dz, dy, dx). dz is the distance between slice centres,
        calculated from the slice positions, not SliceThickness.

    Example
    -------
    >>> ct, ct_spacing = load_series("data/raw/CT/Helical")
    >>> ct.shape, ct_spacing
    ((522, 512, 512), (0.625, 0.586, 0.586))
    """
    folder = Path(folder)

    # --- Check inputs ---
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # 1. Load and sort the slices with the toolbox loader (the rescale is applied there)
    volume, first_ds, slices = load_dicom(folder, series_uid)
    volume = volume.astype(np.float32)                       # Half the memory of float64, enough precision

    # 2. In-plane spacing (dy, dx)
    dy, dx = get_pixel_spacing(first_ds)

    # 3. Slice spacing (dz) from the slice positions
    if len(slices) > 1:
        dz = slice_spacing_mm(slices)
    else:
        dz = float(first_ds.get("SliceThickness", np.nan))   # Only one slice: use SliceThickness if available

    return volume, (dz, dy, dx)


def load_image(file: str | Path):
    """
    Load a single 2D image (ultrasound or X-ray) and its pixel spacing.

    Unlike load_dicom, this also handles colour ultrasound images (converted to
    grayscale) and X-ray images stored as MONOCHROME1 (inverted, so that high
    attenuation is bright), and reads the X-ray and ultrasound pixel spacing.

    Parameters
    ----------
    file : str or Path
        DICOM file.

    Returns
    -------
    image : np.ndarray
        2D grayscale image (rows, cols). Multi-frame files are returned as (frames, rows, cols).
    spacing : tuple or None
        Pixel spacing in mm (row spacing, column spacing), or None if not in the header.

    Example
    -------
    >>> xray, xray_spacing = load_image("data/raw/XRay/image.dcm")
    >>> xray.shape, xray_spacing
    ((3032, 2520), (0.139, 0.139))
    """
    file = Path(file)

    # --- Check inputs ---
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")

    # 1. Read the DICOM file
    ds = pydicom.dcmread(file)

    # 2. Get the pixel array
    image = ds.pixel_array.astype(np.float32)

    # 3. Convert colour (RGB) images to grayscale (common for ultrasound)
    if ds.get("SamplesPerPixel", 1) == 3:
        image = color.rgb2gray(image)

    # 4. Apply the rescale, if present
    slope     = float(ds.get("RescaleSlope", 1))            # 1 if not given
    intercept = float(ds.get("RescaleIntercept", 0))        # 0 if not given
    image     = image * slope + intercept

    # 5. MONOCHROME1 shows high values as dark: invert, so high attenuation is bright (common for X-ray)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image

    # 6. Pixel spacing (handles X-ray and ultrasound headers)
    spacing = get_pixel_spacing(ds)

    return image, spacing

# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str):
    """
    Save a DataFrame as a CSV file in results/.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    name : str
        File name without extension.

    Example
    -------
    >>> save_table(summary, "mri_objects")
    Saved table: results/mri_objects.csv
    """
    # --- Check inputs ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, but got {type(df)}.")

    RESULTS.mkdir(parents=True, exist_ok=True)              # Create results/ if it doesn't exist
    path = RESULTS / f"{name}.csv"                          # Full path to the CSV file
    df.to_csv(path, index=False)                            # Save without the index column
    print(f"Saved table: {path.relative_to(PROJECT)}")


def save_figure(fig: plt.Figure, name: str, dpi: int = 300):
    """
    Save a matplotlib figure as a PNG file in images/.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save.
    name : str
        File name without extension.
    dpi : int
        Resolution in dots per inch. Default is 300 (good for reports).

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> save_figure(fig, "ct_histogram")
    Saved figure: images/ct_histogram.png
    """
    IMAGES.mkdir(parents=True, exist_ok=True)               # Create images/ if it doesn't exist
    path = IMAGES / f"{name}.png"                           # Full path to the PNG file
    fig.savefig(path, dpi=dpi, bbox_inches="tight")         # Tight borders, no extra white space
    print(f"Saved figure: {path.relative_to(PROJECT)}")