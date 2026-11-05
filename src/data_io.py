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
IMAGES = PROJECT / "images"

# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def get_pixel_spacing(ds):
    """
    Get pixel spacing in mm from a DICOM header (CT, MRI, X-ray or US).

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        DICOM header.

    Returns
    -------
    tuple or None
        Pixel spacing in mm (dy, dx), or None if not found.
    """
    # --- Check inputs ---
    if not isinstance(ds, pydicom.dataset.Dataset):
        raise TypeError(f"Input must be a pydicom Dataset, but got {type(ds)}.")
    
    # 1. Try PixelSpacing (CT, MRI)
    if "PixelSpacing" in ds:
        dy, dx = ds.PixelSpacing

    # 2. Else try ImagerPixelSpacing (X-ray)
    elif "ImagerPixelSpacing" in ds:
        dy, dx = ds.ImagerPixelSpacing

    # 3. Else try SequenceOfUltrasoundRegions (US), given in cm
    elif "SequenceOfUltrasoundRegions" in ds:
        region = ds.SequenceOfUltrasoundRegions[0]
        # Get the physical delta values and convert from cm to mm
        dy = float(region.PhysicalDeltaY) * 10      
        dx = float(region.PhysicalDeltaX) * 10 

    # No spacing information found -> return None
    else:
        return None                                 
    # 4. Convert to floats and return (dy, dx)
    return float(dy), float(dx)

def _slice_position(ds):
    """
    Position of a slice along the scan direction (mm).
 
    Projects ImagePositionPatient onto the slice normal, which is the
    cross product of the row and column directions in ImageOrientationPatient.
    This works for any scan orientation (axial, sagittal, coronal).

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        DICOM header of a slice.    
    
    Returns
    -------
    float
        Position of the slice along the scan direction (mm).
    """
    orientation = np.array(ds.ImageOrientationPatient, dtype=float) # Define the orientation vector
    row_dir, col_dir = orientation[:3], orientation[3:]             # Split into row and column direction vectors
    normal = np.cross(row_dir, col_dir)                             # Compute the normal vector to the slice plane
    position = np.array(ds.ImagePositionPatient, dtype=float)       # Get the position of the slice in 3D space

    # Get the position along the normal vector (dot product)
    position_along_normal = np.dot(position, normal)       
    pos_along_normal      = float(position_along_normal)            # Convert to float for consistency         
    return pos_along_normal

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_series(folder: str | Path):
    """
    Loads CT or MRI series as a 3D array.

    Parameters
    ----------
    folder : str or Path
        Folder containing the DICOM slices of one series.
 
    Returns
    -------
    volume : np.ndarray
        3D array (slices, rows, cols). CT values are in Hounsfield units.
    spacing : tuple
        Voxel spacing in mm (dz, dy, dx). 
    """
    folder = Path(folder)

    # --- Check inputs --
    if not folder.is_dir():
       raise FileNotFoundError(f"Folder not found: {folder}")

    # 1. Read all DICOM files in the folder
    slices = []                                         # Empty list to store DICOM slices
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.name.startswith("."):   # Skip hidden files and directories
            continue
        try:
            ds = pydicom.dcmread(f)                     # Read the DICOM file and save as a pydicom Dataset
        except pydicom.errors.InvalidDicomError:        
            continue                                    # Not a DICOM file, skip
        if "PixelData" in ds and "ImagePositionPatient" in ds:
            slices.append(ds)                           # Keep only image slices

    if len(slices) == 0:
        raise ValueError(f"No DICOM image slices found in {folder}")

    # 2. Sort slices by their position along the scan direction
    slices.sort(key=_slice_position)
 
    # 3. Stack pixel arrays into one 3D array
    volume = np.stack([ds.pixel_array for ds in slices]).astype(np.float32)
 
    # 4. CT: convert stored values to Hounsfield units (per slice)
    if slices[0].Modality == "CT":
        for i, ds in enumerate(slices):
            slope = float(ds.get("RescaleSlope", 1))          # Get the rescale slope 
            intercept = float(ds.get("RescaleIntercept", 0))  # Get the rescale intercept
            volume[i] = volume[i] * slope + intercept         # Convert to Hounsfield units (HU)
 
    # 5. In-plane spacing (dy, dx)
    dy, dx = get_pixel_spacing(slices[0])
 
    # 6. Slice spacing (dz) from the distance between slice positions
    if len(slices) > 1:
        positions = np.array([_slice_position(ds) for ds in slices])  # Get the positions of all slices along the scan direction
        dz = float(np.median(np.diff(positions)))                     # Compute the median slice spacing (dz)
    else:
        dz = float(slices[0].get("SliceThickness", np.nan))           # If only one slice, use SliceThickness if available, else NaN
 
    return volume, (dz, dy, dx)


def load_image(file: str | Path):
    """
    Load a single 2D image (US or X-ray).
 
    Parameters
    ----------
    file : str or Path
        DICOM file.
 
    Returns
    -------
    image : np.ndarray
        2D grayscale image (rows, cols). Multi-frame files are returned as (frames, rows, cols).
    spacing : tuple or None
        Pixel spacing in mm (dy, dx), or None if not in the header.
    """
    file = Path(file)

    # --- Check inputs --
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")
 
    # 1. Read the DICOM file
    ds = pydicom.dcmread(file)
 
    # 2. Get the pixel array
    image = ds.pixel_array.astype(np.float32)
 
    # 3. Convert RGB images to grayscale
    if ds.get("SamplesPerPixel", 1) == 3:
        image = color.rgb2gray(image)
 
    # 4. Apply rescale if present (some X-ray files use it)
    slope     = float(ds.get("RescaleSlope", 1))        # Get the rescale slope (default 1 if not present)
    intercept = float(ds.get("RescaleIntercept", 0))    # Get the rescale intercept (default 0 if not present)
    image     = image * slope + intercept               # Apply the rescale to convert to physical units (HU for CT, Intensity for X-ray)
 
    # 5. Invert MONOCHROME1 so that higher values are displayed as brighter (common for X-ray)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image                     
 
    # 6. Pixel spacing
    spacing = get_pixel_spacing(ds)
 
    return image, spacing

# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str):
    """
    Save a DataFrame as CSV in results/.
 
    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    name : str
        File name without extension.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, but got {type(df)}.")
 
    RESULTS.mkdir(parents=True, exist_ok=True)          # Create the results folder if it doesn't exist
    path = RESULTS / f"{name}.csv"                      # Save the DataFrame as CSV
    df.to_csv(path, index=False)                        # Save without the index column
    print(f"Saved table: {path.relative_to(PROJECT)}")  
 
 
def save_figure(fig: plt.Figure, name: str, dpi: int=300):
    """
    Save a matplotlib figure as PNG in images/.
 
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name without extension.
    dpi : int
        Resolution in dots per inch.
        Default is *300*.
    """
    IMAGES.mkdir(parents=True, exist_ok=True)            # Create the images folder if it doesn't exist
    path = IMAGES / f"{name}.png"                        # Save the figure as PNG  
    fig.savefig(path, dpi=dpi, bbox_inches="tight")      # Save the figure with tight bounding box
    print(f"Saved figure: {path.relative_to(PROJECT)}")     