# FILE: html_viewer.py
# DESCRIPTION: Interactive 3-plane viewer in JavaScript, so it also works in HTML exports.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import base64
import json
import uuid

import numpy as np
from IPython.display import HTML, display

from .viewers import CT_WINDOWS                     # Same HU windows as the Python viewer

_HTML_TEMPLATE = """
<div id="__ID__" style="font-family: sans-serif; font-size: 13px; padding: 6px;">
  <div class="info" style="margin-bottom: 6px;"></div>
  <div style="display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap;">
    <div><div class="lab"></div><canvas style="background: black;"></canvas></div>
    <div><div class="lab"></div><canvas style="background: black;"></canvas></div>
    <div><div class="lab"></div><canvas style="background: black;"></canvas></div>
  </div>
  <div style="margin-top: 8px; padding: 6px; border: 1px solid #ccc; width: 380px;">
    <div style="margin-bottom: 4px;">Window (__UNIT__) <select></select></div>
    <div><span style="display: inline-block; width: 70px;" class="name"></span>
         <input type="range" style="width: 220px;"> <span class="val"></span></div>
    <div><span style="display: inline-block; width: 70px;" class="name"></span>
         <input type="range" style="width: 220px;"> <span class="val"></span></div>
    <div><span style="display: inline-block; width: 70px;" class="name"></span>
         <input type="range" style="width: 220px;"> <span class="val"></span></div>
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
  const [Z, Y, X] = cfg.shape;
  const [dz, dy, dx] = cfg.spacing;
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

  // --- Slider ranges and window options ---
  [Z, Y, X].forEach((n, i) => {
    sliders[i].min = 0; sliders[i].max = n - 1; sliders[i].value = Math.floor(n / 2);
    names[i].textContent = cfg.names[i];
  });
  Object.keys(cfg.windows).forEach(k => {
    const o = document.createElement("option"); o.value = k; o.textContent = k; select.appendChild(o);
  });

  // --- Same mm scale in all views: (rows, cols, row spacing, col spacing) ---
  const views = [[Y, X, dy, dx], [Z, X, dz, dx], [Z, Y, dz, dy]];
  const scale = cfg.height_px / Math.max(Y * dy, Z * dz);         // px per mm
  views.forEach(([R, C, sr, sc], i) => {
    canvases[i].width = C; canvases[i].height = R;
    canvases[i].style.width = (C * sc * scale) + "px";
    canvases[i].style.height = (R * sr * scale) + "px";
  });

  function render(canvas, R, C, get, grey, hLine, vLine) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(C, R);
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const g = grey(get(r, c));
        const k = 4 * (r * C + c);
        img.data[k] = img.data[k + 1] = img.data[k + 2] = g;
        img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.fillStyle = "rgba(255, 0, 0, 0.8)";                      // Crosshairs
    ctx.fillRect(0, hLine, C, 1);
    ctx.fillRect(vLine, 0, 1, R);
  }

  function draw() {
    const z = +sliders[0].value, y = +sliders[1].value, x = +sliders[2].value;
    const [wlo, whi] = cfg.windows[select.value];
    const qlo = toQ(wlo), qhi = toQ(whi);
    const grey = q => { let t = (q - qlo) / (qhi - qlo); return 255 * (t < 0 ? 0 : t > 1 ? 1 : t); };

    // View 0: vol[z, :, :]; views 1 and 2 are flipped so z increases upwards
    render(canvases[0], Y, X, (r, c) => vol[(z * Y + r) * X + c], grey, y, x);
    render(canvases[1], Z, X, (r, c) => vol[((Z - 1 - r) * Y + y) * X + c], grey, Z - 1 - z, x);
    render(canvases[2], Z, Y, (r, c) => vol[((Z - 1 - r) * Y + c) * X + x], grey, Z - 1 - z, y);

    const pos = [[z, Z, dz], [y, Y, dy], [x, X, dx]];
    pos.forEach(([p, n, s], i) => {
      labels[i].textContent = `${cfg.names[i]} — ${p}/${n - 1} (${(p * s).toFixed(1)} mm)`;
      vals[i].textContent = p;
    });
    info.innerHTML = `<b>${cfg.title}</b> | value at crosshair: ` +
                     `${toValue(vol[(z * Y + y) * X + x]).toFixed(0)} ${cfg.unit}` +
                     (cfg.step > 1 ? ` | downsampled by ${cfg.step}` : "");
  }

  sliders.forEach(s => s.addEventListener("input", draw));
  select.addEventListener("change", draw);
  draw();
})();
</script>
"""


def html_viewer(vol: np.ndarray, spacing: tuple, names=("Axial", "Coronal", "Sagittal"),
                title: str = "", modality: str = "CT", max_voxels: int = 3_000_000,
                height_px: int = 300):
    """
    Interactive 3-plane viewer that runs in the browser (JavaScript), so the
    sliders also work in HTML exports of the notebook.

    The volume is embedded in the notebook output, so large volumes make the
    notebook file large. Volumes with more than `max_voxels` voxels are
    downsampled (every 2nd, 3rd, ... voxel) and the spacing is adjusted.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (slices, rows, cols).
    spacing : tuple
        Voxel spacing in mm (dz, dy, dx).
    names : tuple of str
        Plane names for the three views.
    title : str
        Title shown above the views.
    modality : str
        "CT" for HU windows, anything else for percentile limits.
    max_voxels : int
        Maximum number of embedded voxels (default 3 million, ~8 MB in the notebook).
    height_px : int
        Display height of the tallest view in pixels.
    """
    if vol.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got shape {vol.shape}.")

    # --- Downsample if the volume is too large to embed ---
    step = 1
    while vol[::step, ::step, ::step].size > max_voxels:
        step += 1
    v = np.asarray(vol[::step, ::step, ::step], dtype=float)
    sp = [float(s) * step for s in spacing]

    # --- Quantise to uint16 (2 bytes per voxel) between the min and max value ---
    lo, hi = float(v.min()), float(v.max())
    q = np.round((v - lo) / ((hi - lo) or 1) * 65535).astype("<u2")
    data = base64.b64encode(q.tobytes()).decode("ascii")

    # --- Display windows ---
    if modality == "CT":
        windows, unit = CT_WINDOWS, "HU"
    else:
        p2, p99 = np.percentile(v, [2, 99])
        windows = {"Auto (2-99 %)": (float(p2), float(p99)), "Full range": (lo, hi)}
        unit = "a.u."

    uid = "hv_" + uuid.uuid4().hex[:10]
    config = {"id": uid, "data": data, "shape": list(v.shape), "spacing": sp,
              "lo": lo, "hi": hi, "windows": {k: list(map(float, w)) for k, w in windows.items()},
              "names": list(names), "title": title, "unit": unit,
              "height_px": height_px, "step": step}

    html = (_HTML_TEMPLATE.replace("__ID__", uid)
                          .replace("__UNIT__", unit)
                          .replace("__CONFIG__", json.dumps(config)))
    display(HTML(html))