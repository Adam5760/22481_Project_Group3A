# FILE: viewers.py
# DESCRIPTION: Functions for displaying images and volumes: 2D slices with axes in mm,
#              outlines of masks, plane names, and an interactive 3-plane viewer.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import base64
import json
import uuid
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Common CT display windows (HU): (min, max)
CT_WINDOWS = {
    "Full (-1000 to 1000)":      (-1000, 1000),
    "Soft tissue (-160 to 240)": (-160, 240),    # level 40, width 400
    "Lung/air (-1350 to 150)":   (-1350, 150),   # level -600, width 1500
    "Bone (-450 to 1050)":       (-450, 1050),   # level 300, width 1500
}

# ---------------------------------------------------------------------------
# 2D display
# ---------------------------------------------------------------------------
def show_image(img: np.ndarray, spacing: tuple | None = None, vmin: float | None = None,
               vmax: float | None = None, title: str = "Image", ax=None, origin: str = "upper"):
    """
    Inspired by show_image from the Lesson 9 Image Analysis Toolbox.

    Display a 2D image, with axes in mm if the pixel spacing is given.

    Compared with the toolbox version, the image can also be drawn in an existing
    subplot (ax), and with row 0 at the bottom (origin="lower"), which is used for
    slices where axis 0 of the volume should point upwards.
    The display limits (vmin, vmax) only change how the image looks, not the values.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    spacing : tuple or None
        (row spacing, column spacing) in mm. Default None shows the axes in pixels.
    vmin, vmax : float or None
        Display limits. Default None uses the image's minimum and maximum.
    title : str
        Title of the subplot. Default is "Image".
    ax : matplotlib Axes or None
        Subplot to draw in. Default None creates a new figure.
    origin : str
        "upper" (row 0 at the top, as in the toolbox) or "lower" (row 0 at the bottom).

    Returns
    -------
    fig : matplotlib Figure
        The figure the image is drawn in.
    ax : matplotlib Axes
        The subplot, for further annotations (lines, rectangles, outlines).

    Example
    -------
    >>> fig, ax = show_image(ct_a3[100], spacing=(0.586, 0.586), vmin=-1000, vmax=1000, title="Slice 100")
    >>> plt.show()
    """
    # --- New figure, or draw in the given subplot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    rows, cols = img.shape

    # --- Axes in pixels ---
    if spacing is None:
        ax.imshow(img, cmap="gray", origin=origin, vmin=vmin, vmax=vmax)
        ax.set(xlabel="Column (pixels)", ylabel="Row (pixels)")

    # --- Axes in mm: the extent places the image in mm instead of pixels ---
    else:
        dy, dx = spacing                                    # DICOM stores row spacing before column spacing
        if origin == "upper":
            extent = [0, cols * dx, rows * dy, 0]           # Row 0 at the top
        else:
            extent = [0, cols * dx, 0, rows * dy]           # Row 0 at the bottom
        ax.imshow(img, cmap="gray", origin=origin, vmin=vmin, vmax=vmax, extent=extent)
        ax.set(xlabel="Horizontal distance (mm)", ylabel="Vertical distance (mm)")

    ax.set_title(title)
    return fig, ax


def contour_mm(ax, mask: np.ndarray, spacing: tuple, **kwargs):
    """
    Draw the outline of a 2D mask on an image shown with show_image in mm.

    Each pixel (row, column) is placed at its centre, ((column + 0.5) * dx, (row + 0.5) * dy) mm,
    so the outline lines up with the image for both origin="upper" and origin="lower".

    Parameters
    ----------
    ax : matplotlib Axes
        Subplot with the image.
    mask : np.ndarray
        2D boolean mask (same shape as the image).
    spacing : tuple
        (row spacing, column spacing) in mm.
    **kwargs
        Passed to matplotlib's contour (e.g. colors="r", linewidths=1).

    Example
    -------
    >>> fig, ax = show_image(ct_a3[100], spacing=(0.586, 0.586), vmin=-1000, vmax=1000)
    >>> contour_mm(ax, obj[100], spacing=(0.586, 0.586), colors="r", linewidths=1)
    >>> plt.show()
    """
    dy, dx = spacing
    x = (np.arange(mask.shape[1]) + 0.5) * dx              # Column centres (mm)
    y = (np.arange(mask.shape[0]) + 0.5) * dy              # Row centres (mm)
    ax.contour(x, y, mask.astype(float), levels=[0.5], **kwargs)   # Line where the mask changes

# ---------------------------------------------------------------------------
# Plane names
# ---------------------------------------------------------------------------
def plane_name(normal) -> str:
    """
    Name of the plane perpendicular to a direction vector.

    The direction is given in patient coordinates (x = left-right, y = front-back,
    z = head-feet). The plane is named after the axis the direction points along most.

    Parameters
    ----------
    normal : array-like
        Direction vector (x, y, z).

    Returns
    -------
    str
        "Sagittal" (normal along x), "Coronal" (along y) or "Axial" (along z).

    Example
    -------
    >>> plane_name([0, 0, 1])
    'Axial'
    """
    axis = np.argmax(np.abs(normal))                        # 0 = x, 1 = y, 2 = z
    return ["Sagittal", "Coronal", "Axial"][axis]


def plane_names(ds: pydicom.dataset.Dataset) -> tuple:
    """
    Plane names for the three views of a volume: vol[i], vol[:, j, :] and vol[:, :, k].

    The names follow from ImageOrientationPatient, so they are correct for any scan
    orientation (e.g. the MRI is acquired sagittally, the CT axially).

    Parameters
    ----------
    ds : pydicom Dataset
        DICOM header of one slice in the series.

    Returns
    -------
    tuple of str
        Plane names for (view 0, view 1, view 2).

    Example
    -------
    >>> plane_names(first_header("data/raw/CT/Helical"))
    ('Axial', 'Coronal', 'Sagittal')
    """
    # --- Check inputs ---
    if not isinstance(ds, pydicom.dataset.Dataset):
        raise TypeError(f"Input must be a pydicom Dataset, but got {type(ds)}.")

    orientation = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = orientation[:3], orientation[3:]    # Along a row, down a column
    normal = np.cross(row_dir, col_dir)                    # Normal of the acquired slices

    # View 0 is perpendicular to the normal, view 1 to the column direction, view 2 to the row direction
    return plane_name(normal), plane_name(col_dir), plane_name(row_dir)


def first_header(folder: str | Path) -> pydicom.dataset.Dataset:
    """
    DICOM header (without pixel data) of the first .dcm file in a folder.

    Faster than loading the whole series when only header information is needed,
    e.g. for plane_names or acquisition parameters.

    Parameters
    ----------
    folder : str or Path
        Folder containing DICOM files.

    Returns
    -------
    pydicom Dataset
        DICOM header.

    Example
    -------
    >>> first_header("data/raw/MRI/3D_T1w").RepetitionTime
    999.0
    """
    files = sorted(Path(folder).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {folder}")
    return pydicom.dcmread(files[0], stop_before_pixels=True)   # Header only, no pixel data

# ---------------------------------------------------------------------------
# Interactive 3-plane viewer (runs in the browser, so it also works in HTML exports)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """
<div id="__ID__" style="font-family: sans-serif; font-size: 13px; padding: 6px;">
  <div class="info" style="margin-bottom: 6px;"></div>
  <!-- Layout: view 0 on the left, views 1 and 2 with the readout and controls on the right.
       If view 0 is not the tallest view, it is moved into the row with views 1 and 2. -->
  <div style="display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap;">
    <div class="view0"><div class="lab" style="margin-bottom: 4px;"></div><canvas></canvas></div>
    <div style="display: flex; flex-direction: column; gap: 8px;">
      <div class="row12" style="display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap;">
        <div><div class="lab" style="margin-bottom: 4px;"></div><canvas></canvas></div>
        <div><div class="lab" style="margin-bottom: 4px;"></div><canvas></canvas></div>
      </div>
      <div class="hover" style="min-height: 1.3em; font-family: monospace;">Hover over a view to read positions</div>
      <div style="padding: 6px; border: 1px solid #ccc; width: 520px;">
        <div style="margin-bottom: 4px;">Window (__UNIT__) <select></select></div>
        <div><span style="display: inline-block; width: 115px;" class="name"></span>
             <input type="range" style="width: 220px;"> <span class="val"></span></div>
        <div><span style="display: inline-block; width: 115px;" class="name"></span>
             <input type="range" style="width: 220px;"> <span class="val"></span></div>
        <div><span style="display: inline-block; width: 115px;" class="name"></span>
             <input type="range" style="width: 220px;"> <span class="val"></span></div>
      </div>
    </div>
  </div>
</div>
<script>
(function () {
  const cfg = __CONFIG__;

  // --- Decode the volume (uint16, quantised between cfg.lo and cfg.hi) ---
  const bin = atob(cfg.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const vol = new Uint16Array(bytes.buffer);
  const [Z, Y, X] = cfg.shape;                      // Shape after downsampling
  const [dz, dy, dx] = cfg.spacing;                 // Spacing after downsampling (mm)
  const step = cfg.step;                            // Downsampling factor
  const toValue = q => cfg.lo + q / 65535 * (cfg.hi - cfg.lo);
  const toQ = v => (v - cfg.lo) / (cfg.hi - cfg.lo) * 65535;

  // --- Elements ---
  const root = document.getElementById(cfg.id);
  const canvases = root.querySelectorAll("canvas");
  const labels = root.querySelectorAll(".lab");
  const sliders = root.querySelectorAll("input[type=range]");
  const names = root.querySelectorAll(".name");
  const vals = root.querySelectorAll(".val");
  const select = root.querySelector("select");
  const info = root.querySelector(".info");
  const hover = root.querySelector(".hover");

  // --- Slider ranges and window options ---
  [Z, Y, X].forEach((n, i) => {
    sliders[i].min = 0; sliders[i].max = n - 1; sliders[i].value = Math.floor(n / 2);
    names[i].textContent = `${cfg.names[i]} (axis ${i})`;
  });
  Object.keys(cfg.windows).forEach(k => {
    const o = document.createElement("option"); o.value = k; o.textContent = k; select.appendChild(o);
  });

  // --- Views: rows, columns, row and column spacing, axis numbers (horizontal, vertical), row 0 at top? ---
  const views = [
    {R: Y, C: X, sr: dy, sc: dx, h: 2, v: 1, top: true},    // View 0: vol[i, :, :]
    {R: Z, C: X, sr: dz, sc: dx, h: 2, v: 0, top: false},   // View 1: vol[:, j, :], axis 0 points up
    {R: Z, C: Y, sr: dz, sc: dy, h: 1, v: 0, top: false},   // View 2: vol[:, :, k], axis 0 points up
  ];
  const scale = cfg.height_px / Math.max(Y * dy, Z * dz);   // Screen pixels per mm (same in all views)

  // --- Layout: if view 0 is lower than views 1 and 2, show all three views in one row ---
  if (Y * dy < Z * dz) {
    const row12 = root.querySelector(".row12");
    row12.insertBefore(root.querySelector(".view0"), row12.firstChild);
  }
  const M = {left: 42, right: 8, top: 6, bottom: 34};       // Margins for the axes

  views.forEach((vw, i) => {
    vw.W = vw.C * vw.sc * scale;                            // Image size on screen (px)
    vw.H = vw.R * vw.sr * scale;
    vw.off = document.createElement("canvas");              // Off-screen image at voxel resolution
    vw.off.width = vw.C; vw.off.height = vw.R;
    canvases[i].width = Math.round(vw.W + M.left + M.right);
    canvases[i].height = Math.round(vw.H + M.top + M.bottom);
  });

  // Tick spacing in mm: 5, 10, 20 or 50, giving about 4-8 ticks
  function tickStep(lengthMm) {
    for (const s of [5, 10, 20, 50, 100]) if (lengthMm / s <= 8) return s;
    return 100;
  }

  function render(i, get, grey, hLine, vLine) {
    const vw = views[i];

    // 1. Image at voxel resolution on the off-screen canvas
    const octx = vw.off.getContext("2d");
    const img = octx.createImageData(vw.C, vw.R);
    for (let r = 0; r < vw.R; r++) {
      for (let c = 0; c < vw.C; c++) {
        const g = grey(get(r, c));
        const k = 4 * (r * vw.C + c);
        img.data[k] = img.data[k + 1] = img.data[k + 2] = g;
        img.data[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    // 2. Scale it onto the visible canvas, inside the margins
    const ctx = canvases[i].getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvases[i].width, canvases[i].height);
    ctx.drawImage(vw.off, M.left, M.top, vw.W, vw.H);

    // 3. Crosshairs (through the centres of the selected voxels)
    ctx.fillStyle = "rgba(255, 0, 0, 0.8)";
    ctx.fillRect(M.left, M.top + (hLine + 0.5) / vw.R * vw.H, vw.W, 1);
    ctx.fillRect(M.left + (vLine + 0.5) / vw.C * vw.W, M.top, 1, vw.H);

    // 4. Axes in mm with ticks and labels
    ctx.strokeStyle = "black"; ctx.fillStyle = "black"; ctx.font = "10px sans-serif";
    ctx.strokeRect(M.left, M.top, vw.W, vw.H);
    const wMm = vw.C * vw.sc, hMm = vw.R * vw.sr;            // Size of the view in mm
    const sx = tickStep(wMm), sy = tickStep(hMm);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let t = 0; t <= wMm + 1e-6; t += sx) {             // Horizontal axis: 0 at the left
      const px = M.left + t * scale;
      ctx.fillRect(px, M.top + vw.H, 1, 4);
      ctx.fillText(t.toFixed(0), px, M.top + vw.H + 5);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let t = 0; t <= hMm + 1e-6; t += sy) {             // Vertical axis: 0 at the top (view 0) or bottom
      const py = vw.top ? M.top + t * scale : M.top + vw.H - t * scale;
      ctx.fillRect(M.left - 4, py, 4, 1);
      ctx.fillText(t.toFixed(0), M.left - 6, py);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`axis ${vw.h} (mm)`, M.left + vw.W / 2, canvases[i].height - 1);
    ctx.save();
    ctx.translate(10, M.top + vw.H / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(`axis ${vw.v} (mm)`, 0, 0);
    ctx.restore();
  }

  function draw() {
    const z = +sliders[0].value, y = +sliders[1].value, x = +sliders[2].value;
    const [wlo, whi] = cfg.windows[select.value];
    const qlo = toQ(wlo), qhi = toQ(whi);
    const grey = q => { let t = (q - qlo) / (qhi - qlo); return 255 * (t < 0 ? 0 : t > 1 ? 1 : t); };

    // View 0: vol[z, :, :]; views 1 and 2 are flipped so axis 0 increases upwards
    render(0, (r, c) => vol[(z * Y + r) * X + c], grey, y, x);
    render(1, (r, c) => vol[((Z - 1 - r) * Y + y) * X + c], grey, Z - 1 - z, x);
    render(2, (r, c) => vol[((Z - 1 - r) * Y + c) * X + x], grey, Z - 1 - z, y);

    // Position of each slider: slice number in the ORIGINAL volume, and mm
    const pos = [[z, dz], [y, dy], [x, dx]];
    pos.forEach(([p, s], i) => {
      const index = p * step;                              // Undo the downsampling
      const mm = (p * s).toFixed(1);
      labels[i].textContent = `${cfg.names[i]}: slice ${index}/${cfg.orig_shape[i] - 1}, ${mm} mm`;
      vals[i].textContent = `slice ${index}, ${mm} mm`;
    });
    info.innerHTML = `<b>${cfg.title}</b> | value at crosshair: ` +
                     `${toValue(vol[(z * Y + y) * X + x]).toFixed(0)} ${cfg.unit}` +
                     (step > 1 ? ` | downsampled by ${step} (slice numbers refer to the full volume)` : "");
  }

  // --- Hover readout: position (slice number and mm) and value under the mouse ---
  canvases.forEach((cv, i) => {
    const vw = views[i];
    cv.addEventListener("mousemove", ev => {
      const rect = cv.getBoundingClientRect();
      const mx = (ev.clientX - rect.left) * cv.width / rect.width;     // Mouse position in canvas pixels
      const my = (ev.clientY - rect.top) * cv.height / rect.height;
      const c = Math.floor((mx - M.left) / vw.W * vw.C);               // Column and row in the view (voxels)
      const r = Math.floor((my - M.top) / vw.H * vw.R);
      if (c < 0 || r < 0 || c >= vw.C || r >= vw.R) { hover.textContent = "Hover over a view to read positions"; return; }

      // Voxel under the mouse (downsampled indices); views 1 and 2 have axis 0 pointing up
      let z = +sliders[0].value, y = +sliders[1].value, x = +sliders[2].value;
      if (i === 0) { y = r; x = c; }
      if (i === 1) { z = Z - 1 - r; x = c; }
      if (i === 2) { z = Z - 1 - r; y = c; }

      const idx = {0: z, 1: y, 2: x};
      const sp  = {0: dz, 1: dy, 2: dx};
      const fmt = a => `axis ${a}: ${(idx[a] * sp[a]).toFixed(1)} mm (slice ${idx[a] * step})`;
      const value = toValue(vol[(z * Y + y) * X + x]).toFixed(0);
      hover.textContent = `${fmt(vw.h)}   ${fmt(vw.v)}   value: ${value} ${cfg.unit}`;
    });
    cv.addEventListener("mouseleave", () => { hover.textContent = "Hover over a view to read positions"; });
  });

  sliders.forEach(s => s.addEventListener("input", draw));
  select.addEventListener("change", draw);
  draw();
})();
</script>
"""


def viewer(vol: np.ndarray, spacing: tuple, names: tuple = ("Axial", "Coronal", "Sagittal"),
           title: str = "", modality: str = "CT", max_voxels: int = 3_000_000, height_px: int = 400):
    """
    Interactive 3-plane viewer with crosshairs, sliders and display windows.

    The viewer runs in the browser (JavaScript), so the sliders also work in HTML exports
    of the notebook and need no running Python kernel. All three views use the same mm
    scale. CT uses fixed HU windows; other modalities (e.g. MRI) use percentile limits.

    Each view has axes in mm, labelled with the volume axis they show (axis 0, 1 or 2).
    The sliders show both the slice number and the position in mm, and hovering over a
    view shows the position and value under the mouse. The slice numbers always
    refer to the full (not downsampled) volume, so they can be used directly for indexing,
    e.g. vol[54] or vol[:, 108, :].

    The volume is embedded in the notebook output, so large volumes make the notebook file
    large. Volumes with more than max_voxels voxels are downsampled (every 2nd, 3rd, ... voxel),
    the spacing is adjusted, and the title says so.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (axis 0, axis 1, axis 2).
    spacing : tuple
        Voxel spacing in mm (axis 0, axis 1, axis 2).
    names : tuple of str
        Plane names for the three views. Use plane_names(ds) to get them from a header.
        Default is ("Axial", "Coronal", "Sagittal").
    title : str
        Title shown above the views.
    modality : str
        "CT" for HU windows, anything else for percentile limits. Default is "CT".
    max_voxels : int
        Maximum number of embedded voxels. Default is 3 million (about 8 MB in the notebook).
    height_px : int
        Display height of the tallest view in pixels. Default is 400.

    Example
    -------
    >>> viewer(ct_a3, ct_spacing, names=ct_names, title="CT — phantom A3", modality="CT")
    >>> viewer(t2_a3, t2_spacing, names=t2_names, title="MRI T2w — phantom A3", modality="MR")
    """
    # --- Check inputs ---
    if vol.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got shape {vol.shape}.")
    if len(spacing) != 3:
        raise ValueError(f"Spacing must have 3 values, but got {spacing}.")
    if len(names) != 3:
        raise ValueError(f"Names must be 3 strings, but got {names}.")

    # --- Downsample if the volume is too large to embed ---
    step = 1
    while vol[::step, ::step, ::step].size > max_voxels:
        step += 1                                           # Keep every 2nd, 3rd, ... voxel
    v = np.asarray(vol[::step, ::step, ::step], dtype=float)
    sp = [float(s) * step for s in spacing]                 # Larger voxels after downsampling

    # --- Store the volume compactly: 16-bit integers between the min and max value ---
    lo, hi = float(v.min()), float(v.max())
    q = np.round((v - lo) / ((hi - lo) or 1) * 65535).astype("<u2")
    data = base64.b64encode(q.tobytes()).decode("ascii")    # Text, so it can be embedded in HTML

    # --- Display windows ---
    if modality == "CT":
        windows, unit = CT_WINDOWS, "HU"                    # Fixed HU windows
    else:
        p2, p99 = np.percentile(v, [2, 99])                 # Uncalibrated values: use percentiles
        if p99 <= p2:                                       # Almost uniform volume: use the full range
            p2, p99 = lo, hi
        windows = {"Auto (2-99 %)": (float(p2), float(p99)), "Full range": (lo, hi)}
        unit = "a.u."

    # --- Settings passed to the JavaScript ---
    uid = "hv_" + uuid.uuid4().hex[:10]                     # Unique id, so several viewers can be shown
    config = {"id": uid, "data": data, "shape": list(v.shape), "spacing": sp,
              "lo": lo, "hi": hi, "windows": {k: list(map(float, w)) for k, w in windows.items()},
              "names": list(names), "title": title, "unit": unit,
              "height_px": height_px, "step": step,
              "orig_shape": list(vol.shape)}                 # Slice numbers are shown for the full volume

    # --- Fill in the HTML template and show it ---
    html = (_HTML_TEMPLATE.replace("__ID__", uid)
                          .replace("__UNIT__", unit)
                          .replace("__CONFIG__", json.dumps(config)))
    display(HTML(html))