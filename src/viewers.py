# FILE: viewers.py
# DESCRIPTION: Interactive viewers and helpers for displaying CT and MRI volumes.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from ipywidgets import IntSlider, Dropdown, Layout, VBox, interactive_output
from IPython.display import display
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Common CT windows (HU): (min, max)
CT_WINDOWS = {
    "Full (-1000 to 1000)":      (-1000, 1000),
    "Soft tissue (-160 to 240)": (-160, 240),    # level 40, width 400
    "Lung/air (-1350 to 150)":   (-1350, 150),   # level -600, width 1500
    "Bone (-450 to 1050)":       (-450, 1050),   # level 300, width 1500
}

# ---------------------------------------------------------------------------
# Plane names
# ---------------------------------------------------------------------------
def plane_name(normal):
    """
    Name of the plane perpendicular to a direction vector.

    Parameters
    ----------
    normal : array-like
        Direction vector (x, y, z) in patient coordinates.

    Returns
    -------
    str
        "Sagittal" (normal along x), "Coronal" (along y) or "Axial" (along z).
    """
    axis = np.argmax(np.abs(normal))                    # 0 = x, 1 = y, 2 = z
    return ["Sagittal", "Coronal", "Axial"][axis]


def plane_names(ds):
    """
    Plane names for the three views vol[z], vol[:, y, :] and vol[:, :, x].

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        DICOM header of one slice in the series.

    Returns
    -------
    tuple of str
        Plane names for (view 0, view 1, view 2).
    """
    if not isinstance(ds, pydicom.dataset.Dataset):
        raise TypeError(f"Input must be a pydicom Dataset, but got {type(ds)}.")

    orientation = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = orientation[:3], orientation[3:]  # Along a row, down a column
    normal = np.cross(row_dir, col_dir)                  # Normal of the acquired slices

    return plane_name(normal), plane_name(col_dir), plane_name(row_dir)

def first_header(folder):
    """
    DICOM header (without pixel data) of the first .dcm file in a folder.

    Parameters
    ----------
    folder : str or Path
        Folder containing DICOM files.

    Returns
    -------
    pydicom.dataset.Dataset
        DICOM header.
    """
    files = sorted(Path(folder).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {folder}")
    return pydicom.dcmread(files[0], stop_before_pixels=True)

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
def viewer(vol: np.ndarray, spacing: tuple, names=("Axial", "Coronal", "Sagittal"), title: str="", modality: str="CT"):
    """
    Interactive 3-plane viewer with crosshairs.
 
    CT uses fixed HU windows, other modalities (e.g. MRI) use percentile limits.
 
    Parameters
    ----------
    vol : np.ndarray
        3D volume (slices, rows, cols).
    spacing : tuple
        Voxel spacing in mm (dz, dy, dx).
    names : tuple of str
        Plane names for the three views. Use plane_names(ds) to get them from a header.
    title : str
        Title shown above the figure.
    modality : str
        "CT" for HU windows, anything else for percentile limits.
    """
    #--- Check inputs ---
    if vol.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got shape {vol.shape}.")
    if len(spacing) != 3:
        raise ValueError(f"Spacing must be a tuple of 3 values (dz, dy, dx), but got {spacing}.")
    if len(names) != 3:
        raise ValueError(f"Names must be a tuple of 3 strings, but got {names}.")
    if not isinstance(title, str):
        raise TypeError(f"Title must be a string, but got {type(title)}.")
    if not isinstance(modality, str):
        raise TypeError(f"Modality must be a string, but got {type(modality)}.")

    # --- Unpack inputs ---
    dz, dy, dx = spacing    # Voxel spacing in mm
    Z, Y, X = vol.shape     # Number of slices, rows and columns
    n0, n1, n2 = names      # Plane names for the three views

    style = {"description_width": "90px", "handle_color": "#9F9E9E", } # Slider style
    layout = Layout(width="100%")                                        # Slider layout

    if modality == "CT":
        windows = CT_WINDOWS
        unit = "HU"
    else:
        lo, hi = np.percentile(vol, [2, 99])
        windows = {"Auto (2-99 %)": (lo, hi),
                   "Full range": (vol.min(), vol.max())}
        unit = "a.u."

    # --- Widgets ---
    z_slider    = IntSlider(Z//2, 0, Z-1, description=n0, style=style, layout=layout)
    y_slider    = IntSlider(Y//2, 0, Y-1, description=n1, style=style, layout=layout)
    x_slider    = IntSlider(X//2, 0, X-1, description=n2, style=style, layout=layout)
    window_dd   = Dropdown(options=list(windows), description=f"Window ({unit})",
                           style=style, layout=Layout(width="290px"))
    
    # --- Plot function ---
    def plot(z: int, y: int, x: int, window: str):
        lo, hi = windows[window]                    # Get window limits
        kw = dict(cmap="gray", vmin=lo, vmax=hi)    # Set colormap and limits
        # Plot the three planes
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ax[0].imshow(vol[z],       aspect=dy/dx, **kw)                      # Axial
        ax[1].imshow(vol[:, y, :], aspect=dz/dx, origin="lower", **kw)      # Coronal
        im = ax[2].imshow(vol[:, :, x], aspect=dz/dy, origin="lower", **kw) # Sagittal
        # Draw crosshairs
        ax[0].axhline(y, c="r", lw=0.6); ax[0].axvline(x, c="r", lw=0.6)
        ax[1].axhline(z, c="r", lw=0.6); ax[1].axvline(x, c="r", lw=0.6)
        ax[2].axhline(z, c="r", lw=0.6); ax[2].axvline(y, c="r", lw=0.6)
        # Set titles
        ax[0].set_title(f"{n0} — {z}/{Z-1}  ({z*dz:.1f} mm)")
        ax[1].set_title(f"{n1} — {y}/{Y-1}  ({y*dy:.1f} mm)")
        ax[2].set_title(f"{n2} — {x}/{X-1}  ({x*dx:.1f} mm)")
        # Hide ticks
        for a in ax:
            a.set_xticks([]); a.set_yticks([])

        fig.colorbar(im, ax=ax, shrink=0.8, label=unit) 
        fig.suptitle(f"{title}  |  value at crosshair: {vol[z, y, x]:.0f} {unit}", x=0.01, ha="left")
        fig.get_layout_engine().set(h_pad=0.15)   # more vertical space between elements (in inches)
        plt.show()
        plt.close(fig)  # Close the figure to avoid duplicate display in Jupyter

    # --- Connect widgets to the plot ---
    out = interactive_output(plot, {"z": z_slider, "y": y_slider, 
                                    "x": x_slider, "window": window_dd})

    # --- Layout ---
    display(VBox([out, VBox([window_dd, z_slider, y_slider, x_slider],
                            layout=Layout(border="1px solid #ccc", width="400px", padding="5px"))],
                            layout=Layout(padding="5px")))