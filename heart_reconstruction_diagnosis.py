"""
Heart Reconstruction and Diagnosis Page
========================================
Two main sections:

  SEGMENTATION & 3D HEART RECONSTRUCTION  (MM-WHS)
    2D U-Net trained on MM-WHS 2017 dataset.
    Segments 7 cardiac structures from CT/MRI NIfTI volumes.
    Generates an interactive 3D mesh viewer with explode / opacity / heatmap controls.
    Checkpoint: mmwhs_ct_best_v1.pth

  DIAGNOSIS
    ├── ACDC — Cardiac Disease Classification
    │   ResBlock U-Net + attention-pooling classification head.
    │   6 pathology classes: Normal, MINF, DCM, HCM, ARV, HHD.
    │   Optional ejection-fraction estimation when ED + ES frames are provided.
    │   Checkpoint: cardiac_v3_best.pth
    │
    └── MyoPS — Myocardial Scar & Edema Detection
        Multi-sequence (C0 / DE / T2) U-Net with ASPP bottleneck + 8-fold TTA.
        Per-region AHA wall-segment localisation and transmurality estimation.
        Checkpoint: myops_best.pth
"""

import io
import os
import tempfile
from collections import defaultdict

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# Pre-trained model paths
# ─────────────────────────────────────────────────────────────────────────────

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_seg_dig")

MMWHS_MODEL = os.path.join(_MODELS_DIR, "mmwhs_ct_best_v1.pth")
ACDC_MODEL  = os.path.join(_MODELS_DIR, "cardiac_v3_best.pth")
MYOPS_MODEL = os.path.join(_MODELS_DIR, "myops_best.pth")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MMWHS_LABEL_MAP = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}
MMWHS_CLASS_NAMES = {
    1: "Myocardium", 2: "Left Ventricle", 3: "Right Ventricle",
    4: "Left Atrium", 5: "Right Atrium", 6: "Ascending Aorta", 7: "Pulmonary Artery",
}
MMWHS_COLORS = {
    1: "#cc3333", 2: "#33bb33", 3: "#3366dd",
    4: "#ddcc22", 5: "#22cccc", 6: "#cc33cc", 7: "#ee8822",
}

ACDC_SEG_CLASSES  = {0: "Background", 1: "RV Cavity", 2: "Myocardium", 3: "LV Cavity"}
ACDC_PATH_CLASSES = {
    0: "Normal", 1: "Infarction (MINF)", 2: "Dilated CM (DCM)",
    3: "Hypertrophic CM (HCM)", 4: "Abnormal RV (ARV)", 5: "Hypertensive HD (HHD)",
}
ACDC_NUM_SEG  = 4
ACDC_NUM_PATH = 6

MYOPS_CLASSES = {
    0: "Background", 1: "LV Cavity", 2: "RV Cavity",
    3: "Myocardium",  4: "Scar",     5: "Edema",
}
MYOPS_NUM_SEG = 6
MYOPS_COLOURS = {
    1: (100, 149, 237), 2: (70, 130, 180),
    3: (210,  80,  80), 4: (255, 200,   0), 5: (0, 220, 220),
}
MYOPS_OPACITY = {1: 0.12, 2: 0.12, 3: 0.18, 4: 0.85, 5: 0.55}

LASC_MODEL   = os.path.join(_MODELS_DIR, "lasc_best.pth")
LASC_CLASSES = {0: "Background", 1: "LA Cavity", 2: "LA Wall"}
LASC_COLORS_RGB = {1: (0.39, 0.58, 0.93), 2: (0.86, 0.31, 0.31)}
LASC_COLORS_HEX = {1: "#6495ED", 2: "#DC4F4F"}
LASC_NUM_CLASSES = 3

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_unlink(path: str) -> None:
    """Delete a temp file, ignoring Windows file-lock errors (WinError 32)."""
    import gc
    try:
        gc.collect()          # encourage nibabel/PyTorch to release handles
        os.unlink(path)
    except PermissionError:
        pass                  # Windows may still hold the handle; OS will clean up on exit
    except FileNotFoundError:
        pass


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _save_nifti_tmp(uploaded_file) -> str:
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.seek(0)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def _percentile_norm(img: np.ndarray) -> np.ndarray:
    fg = img[img > 0] if img.max() > 0 else img.ravel()
    if len(fg) == 0:
        return img.astype(np.float32)
    p1, p99 = np.percentile(fg, [1, 99])
    return np.clip((img - p1) / max(p99 - p1, 1e-8), 0, 1).astype(np.float32)


def _pad32(img, lbl=None):
    import torch.nn.functional as F
    _, _, h, w = img.shape
    nh = ((h + 31) // 32) * 32
    nw = ((w + 31) // 32) * 32
    img = F.pad(img, (0, nw - w, 0, nh - h))
    if lbl is not None:
        lbl = F.pad(lbl, (0, nw - w, 0, nh - h))
    return img, lbl, h, w


def _model_badge(label: str, value: str, color: str = "#1ABC9C") -> str:
    return (
        f"<span style='background:{color}22;border:1px solid {color};"
        f"border-radius:6px;padding:2px 10px;color:{color};"
        f"font-size:0.82rem;margin-right:6px'><b>{label}</b> {value}</span>"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MM-WHS Segmentation & 3D Heart Reconstruction
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading MM-WHS U-Net…")
def _load_mmwhs():
    import torch
    import torch.nn as nn
    device = _device()

    class DC(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            )
        def forward(s, x): return s.net(x)

    class UNet(nn.Module):
        def __init__(s, nc):
            super().__init__()
            s.e1 = DC(1, 32);  s.e2 = DC(32, 64);  s.e3 = DC(64, 128);  s.e4 = DC(128, 256)
            s.pool = nn.MaxPool2d(2); s.drop = nn.Dropout2d(0.1); s.bot = DC(256, 512)
            s.u4 = nn.ConvTranspose2d(512, 256, 2, stride=2); s.d4 = DC(512, 256)
            s.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2); s.d3 = DC(256, 128)
            s.u2 = nn.ConvTranspose2d(128,  64, 2, stride=2); s.d2 = DC(128,  64)
            s.u1 = nn.ConvTranspose2d( 64,  32, 2, stride=2); s.d1 = DC( 64,  32)
            s.out = nn.Conv2d(32, nc, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2))
            e4 = s.e4(s.drop(s.pool(e3))); b = s.bot(s.drop(s.pool(e4)))
            d4 = s.d4(torch.cat([s.u4(b), e4], 1))
            d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
            d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
            d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return s.out(d1)

    model = UNet(8)
    ckpt = torch.load(MMWHS_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval().to(device)
    return model, device, ckpt


def _mmwhs_segment(model, vol: np.ndarray, device: str, is_ct: bool) -> np.ndarray:
    import torch
    CT_WC, CT_WW = 200, 800
    preds = np.zeros(vol.shape, dtype=np.int32)
    for z in range(vol.shape[2]):
        s = vol[:, :, z].astype(np.float32)
        if is_ct:
            vn, vx = CT_WC - CT_WW // 2, CT_WC + CT_WW // 2
            s = np.clip((s - vn) / (vx - vn), 0, 1)
        else:
            s = _percentile_norm(s)
        t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0).float()
        t, _, oh, ow = _pad32(t, None)
        with torch.no_grad():
            p = model(t.to(device)).argmax(1).squeeze().cpu().numpy()
        preds[:, :, z] = p[:vol.shape[0], :vol.shape[1]]
    return preds


def _laplacian_smooth(verts, faces, factor=0.3):
    adj = defaultdict(set)
    for f in faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])
    new_verts = verts.copy()
    for i in range(len(verts)):
        if adj[i]:
            avg = verts[list(adj[i])].mean(axis=0)
            new_verts[i] = verts[i] + factor * (avg - verts[i])
    return new_verts


def _compute_normals(verts, faces):
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def _extract_smooth_mesh(volume, label, sigma=2.5, step=1, spacing=None):
    from skimage import measure
    from scipy import ndimage
    binary = (volume == label).astype(np.float32)
    if binary.sum() < 100:
        return None, None, None
    labeled_arr, n_f = ndimage.label(binary)
    if n_f > 1:
        sizes = ndimage.sum(binary, labeled_arr, range(1, n_f + 1))
        binary = (labeled_arr == np.argmax(sizes) + 1).astype(np.float32)
    if binary.sum() < 50:
        return None, None, None
    smooth = ndimage.gaussian_filter(binary, sigma=sigma)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            smooth, level=0.5, step_size=step, allow_degenerate=False,
            spacing=spacing if spacing else (1., 1., 1.),
        )
    except Exception:
        return None, None, None
    for _ in range(5):
        verts = _laplacian_smooth(verts, faces)
    return verts, faces, _compute_normals(verts, faces)


def _decimate(verts, faces, max_faces=20000):
    if len(faces) <= max_faces:
        return verts, faces
    idx = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return verts, faces[idx]


def _build_heart_html(meshes: dict, patient_id: str) -> str:
    import json
    centers = {}; global_center = np.zeros(3); n = 0
    for c, (verts, _, _) in meshes.items():
        if verts is not None:
            centers[c] = verts.mean(axis=0)
            global_center += centers[c]; n += 1
    if n > 0:
        global_center /= n

    trace_data = []; custom_data = {}
    for c, (verts, faces, _) in meshes.items():
        if verts is None:
            continue
        verts, faces = _decimate(verts, faces)
        color = MMWHS_COLORS[c]; name = MMWHS_CLASS_NAMES[c]
        opacity = 0.3 if c == 1 else 0.85
        center = centers.get(c, global_center)
        direction = center - global_center
        norm = np.linalg.norm(direction)
        direction = (direction / norm) if norm > 0 else np.zeros(3)
        dists = np.linalg.norm(verts - center, axis=1)
        dists_norm = (dists - dists.min()) / max(dists.max() - dists.min(), 1e-8)
        trace_data.append({
            "type": "mesh3d",
            "x": verts[:, 0].tolist(), "y": verts[:, 1].tolist(), "z": verts[:, 2].tolist(),
            "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist(),
            "name": name, "color": color, "opacity": opacity,
            "showlegend": True, "visible": True, "flatshading": False,
            "lighting": {"ambient": 0.35, "diffuse": 0.65, "specular": 0.4,
                         "roughness": 0.3, "fresnel": 0.2},
            "lightposition": {"x": 200, "y": 200, "z": 400},
            "hovertemplate": f"<b>{name}</b><br>x:%{{x:.0f}} y:%{{y:.0f}} z:%{{z:.0f}}<extra></extra>",
        })
        custom_data[str(c)] = {
            "label": c, "explode_dir": direction.tolist(),
            "center": center.tolist(), "intensity_values": dists_norm.tolist(),
        }

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3D Heart – {patient_id}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111;color:#eee;font-family:'Segoe UI',sans-serif;overflow:hidden}}
#plot{{width:100vw;height:100vh}}
#controls{{position:fixed;top:15px;left:15px;z-index:100;background:rgba(20,20,35,.92);
  border-radius:12px;padding:18px;min-width:260px;border:1px solid rgba(255,255,255,.1);
  backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,.5)}}
#controls h2{{font-size:16px;margin-bottom:12px;color:#fff}}
#controls h3{{font-size:12px;color:#888;margin:12px 0 6px;text-transform:uppercase;letter-spacing:1px}}
.slider-row{{display:flex;align-items:center;gap:10px;margin:6px 0}}
.slider-row label{{font-size:13px;min-width:80px}}
.slider-row input[type=range]{{flex:1;accent-color:#66aaff}}
.slider-row .val{{font-size:12px;color:#aaa;min-width:35px;text-align:right}}
.btn{{display:inline-block;padding:7px 14px;margin:3px;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.15);border-radius:6px;color:#ddd;font-size:12px;
  cursor:pointer;transition:all .2s}}
.btn:hover{{background:rgba(255,255,255,.15);color:#fff}}
.btn.active{{background:rgba(100,170,255,.25);border-color:#66aaff;color:#88ccff}}
.struct-row{{display:flex;align-items:center;gap:8px;padding:4px 6px;cursor:pointer;
  border-radius:4px;transition:background .2s}}
.struct-row:hover{{background:rgba(255,255,255,.05)}}
.color-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0}}
.struct-name{{font-size:13px;flex:1}}
.struct-toggle{{font-size:11px;color:#888}}
</style></head><body>
<div id="plot"></div>
<div id="controls">
  <h2>&#x2764; {patient_id}</h2>
  <h3>View Mode</h3>
  <div>
    <span class="btn active" onclick="setMode('solid')" id="btn-solid">Solid</span>
    <span class="btn" onclick="setMode('heatmap')" id="btn-heatmap">Heatmap</span>
    <span class="btn" onclick="setMode('xray')" id="btn-xray">X-Ray</span>
  </div>
  <h3>Controls</h3>
  <div class="slider-row"><label>Explode</label>
    <input type="range" id="explode" min="0" max="100" value="0" oninput="updateExplode(this.value)">
    <span class="val" id="explode-val">0</span></div>
  <div class="slider-row"><label>Opacity</label>
    <input type="range" id="opacity" min="10" max="100" value="85" oninput="updateOpacity(this.value)">
    <span class="val" id="opacity-val">85%</span></div>
  <h3>Structures</h3>
  <div id="struct-list"></div>
  <div style="margin-top:12px">
    <span class="btn" onclick="showAll()">Show All</span>
    <span class="btn" onclick="resetView()">Reset View</span>
  </div>
</div>
<script>
const traces={json.dumps(trace_data)};
const meta={json.dumps(custom_data)};
const origX=traces.map(t=>[...t.x]);
const origY=traces.map(t=>[...t.y]);
const origZ=traces.map(t=>[...t.z]);
const structVisible={{}};
const layout={{
  scene:{{xaxis:{{visible:false}},yaxis:{{visible:false}},zaxis:{{visible:false}},
    aspectmode:'data',bgcolor:'rgb(17,17,22)',camera:{{eye:{{x:1.8,y:.8,z:.6}}}}}},
  paper_bgcolor:'rgb(17,17,22)',margin:{{l:0,r:0,t:0,b:0}},showlegend:false
}};
Plotly.newPlot('plot',traces,layout,{{responsive:true,displayModeBar:false}});
const sl=document.getElementById('struct-list');
const keys=Object.keys(meta);
traces.forEach((t,i)=>{{
  structVisible[i]=true;
  const row=document.createElement('div'); row.className='struct-row';
  row.innerHTML=`<div class="color-dot" style="background:${{t.color}}"></div>
    <span class="struct-name">${{t.name}}</span>
    <span class="struct-toggle" id="tog-${{i}}">ON</span>`;
  row.onclick=()=>toggleStruct(i); sl.appendChild(row);
}});
function toggleStruct(idx){{
  structVisible[idx]=!structVisible[idx];
  Plotly.restyle('plot',{{visible:structVisible[idx]}},[idx]);
  const tog=document.getElementById('tog-'+idx);
  tog.textContent=structVisible[idx]?'ON':'OFF';
  tog.style.color=structVisible[idx]?'#88ff88':'#ff5555';
}}
function showAll(){{
  traces.forEach((_,i)=>{{
    structVisible[i]=true; Plotly.restyle('plot',{{visible:true}},[i]);
    const tog=document.getElementById('tog-'+i);
    tog.textContent='ON'; tog.style.color='#88ff88';
  }});
}}
function updateExplode(val){{
  document.getElementById('explode-val').textContent=val;
  const factor=val/100*40;
  traces.forEach((t,i)=>{{
    const m=meta[keys[i]]; const dir=m.explode_dir;
    Plotly.restyle('plot',{{
      x:[origX[i].map(v=>v+dir[0]*factor)],
      y:[origY[i].map(v=>v+dir[1]*factor)],
      z:[origZ[i].map(v=>v+dir[2]*factor)]
    }},[i]);
  }});
}}
function updateOpacity(val){{
  document.getElementById('opacity-val').textContent=val+'%';
  const op=val/100;
  traces.forEach((t,i)=>{{
    const base=t.name==='Myocardium'?.3:.85;
    Plotly.restyle('plot',{{opacity:Math.min(op,base*(op/.85))}},[i]);
  }});
}}
let currentMode='solid';
function setMode(mode){{
  currentMode=mode;
  document.querySelectorAll('.btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('btn-'+mode).classList.add('active');
  if(mode==='solid'){{
    traces.forEach((t,i)=>{{
      Plotly.restyle('plot',{{opacity:t.name==='Myocardium'?.3:.85,
        colorscale:null,intensity:null,color:t.color,showscale:false}},[i]);
    }});
  }}else if(mode==='heatmap'){{
    traces.forEach((t,i)=>{{
      Plotly.restyle('plot',{{intensity:[meta[keys[i]].intensity_values],
        colorscale:'Portland',color:null,opacity:.9,showscale:i===0}},[i]);
    }});
  }}else if(mode==='xray'){{
    traces.forEach((t,i)=>{{
      Plotly.restyle('plot',{{opacity:.15,colorscale:null,intensity:null,
        color:t.color,showscale:false}},[i]);
    }});
  }}
}}
function resetView(){{
  updateExplode(0); document.getElementById('explode').value=0;
  updateOpacity(85); document.getElementById('opacity').value=85;
  setMode('solid'); showAll();
  Plotly.relayout('plot',{{'scene.camera':{{eye:{{x:1.8,y:.8,z:.6}}}}}});
}}
</script></body></html>"""


def _render_mmwhs_tab():
    st.markdown("""
    <div style='background:#273241;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem'>
    <h4 style='color:#1ABC9C;margin:0'>MM-WHS — Whole Heart Segmentation &amp; 3D Reconstruction</h4>
    <p style='color:#BDC3C7;margin:.4rem 0 0'>
    2D U-Net (32→64→128→256→512 channels) trained on the MM-WHS 2017 dataset.<br>
    Segments 7 cardiac structures from CT or MRI NIfTI volumes and renders an interactive
    3D mesh viewer with explode, opacity and heatmap controls.
    </p></div>""", unsafe_allow_html=True)

    ok = os.path.isfile(MMWHS_MODEL)
    st.markdown(
        f"<p>{_model_badge('mmwhs_ct_best_v1.pth', '✓' if ok else '✗', '#2ECC71' if ok else '#E74C3C')}</p>",
        unsafe_allow_html=True,
    )
    if not ok:
        st.error("Checkpoint not found. Check the model path."); return

    cfg_col, info_col = st.columns([1, 1], gap="large")
    with cfg_col:
        st.markdown("#### Settings")
        modality = st.radio("Modality", ["CT", "MRI"], horizontal=True, key="mmwhs_mod")
        sigma    = st.slider("Gaussian smoothing σ", 1.0, 4.0, 2.5, 0.5, key="mmwhs_sigma",
                             help="Higher = smoother meshes")
        step     = st.select_slider("Marching cubes step", [1, 2], value=1, key="mmwhs_step",
                                    help="Step 2 is faster with slightly lower detail")
        nii_file = st.file_uploader(
            "NIfTI volume (.nii / .nii.gz)", type=["nii", "gz"], key="mmwhs_nii",
            help="Upload a cardiac CT or MRI volume. Pre-labelled volumes are detected automatically.",
        )

    with info_col:
        st.markdown("#### Structures segmented")
        for c, name in MMWHS_CLASS_NAMES.items():
            col_hex = MMWHS_COLORS[c]
            st.markdown(
                f"<span style='color:{col_hex};font-size:1.1rem'>&#9679;</span> **{c}.** {name}",
                unsafe_allow_html=True,
            )

    st.divider()
    run = st.button("▶ Run Segmentation & 3D Reconstruction", type="primary",
                    disabled=nii_file is None, key="mmwhs_run")

    if not nii_file:
        st.info("Upload a NIfTI file to enable reconstruction."); return
    if not run:
        return

    try:
        import nibabel as nib
        import torch  # noqa: F401
    except ImportError as e:
        st.error(f"Missing dependency: {e}. "
                 "Run: `pip install nibabel torch torchvision scikit-image scipy`"); return

    tmp_path = None
    try:
        with st.spinner("Loading model…"):
            model, device, ckpt = _load_mmwhs()

        with st.spinner("Loading NIfTI volume…"):
            tmp_path = _save_nifti_tmp(nii_file)
            nii = nib.load(tmp_path)
            raw_vol = nii.get_fdata()
            spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])

        st.info(
            f"Volume shape: **{raw_vol.shape}** | "
            f"Voxel spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm"
        )

        # Detect whether the file contains raw label values or image intensity.
        # A true MM-WHS label volume contains ONLY the discrete values
        # {0, 205, 420, 500, 550, 600, 820, 850} — nothing else.
        # A raw CT has thousands of distinct HU values, so the subset check fails.
        vol_int = raw_vol.astype(np.int32)
        unique  = np.unique(vol_int)
        _valid_labels = {0, 205, 420, 500, 550, 600, 820, 850}
        if set(unique.tolist()).issubset(_valid_labels) and len(unique) > 1:
            remapped = np.zeros_like(vol_int)
            for o, c in MMWHS_LABEL_MAP.items():
                remapped[vol_int == o] = c
            preds = remapped
            st.success("Pre-segmented NIfTI detected — using label map directly.")
        else:
            with st.spinner(f"Running U-Net segmentation ({raw_vol.shape[2]} slices)…"):
                preds = _mmwhs_segment(model, raw_vol.astype(np.float32), device,
                                       is_ct=(modality == "CT"))
            dice_str = f"{ckpt.get('dice', 0):.4f}" if "dice" in ckpt else "?"
            st.success(
                f"Segmentation done — epoch **{ckpt.get('epoch', '?')}** | "
                f"mDice **{dice_str}** | device **{device.upper()}**"
            )

        # Per-structure voxel counts
        summary_cols = st.columns(len(MMWHS_CLASS_NAMES))
        for i, (c, name) in enumerate(MMWHS_CLASS_NAMES.items()):
            n_vox = int(np.sum(preds == c))
            summary_cols[i].metric(name[:6], f"{n_vox:,}")

        pid = nii_file.name.replace(".nii.gz", "").replace(".nii", "")

        # ── 2D Segmentation Visualisation ────────────────────────────────────
        with st.spinner("Generating 2D segmentation preview…"):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            MMWHS_RGB = {
                1: (204, 51, 51), 2: (51, 187, 51), 3: (51, 102, 221),
                4: (221, 204, 34), 5: (34, 204, 204), 6: (204, 51, 204), 7: (238, 136, 34),
            }
            nz_vol   = raw_vol.shape[2]
            is_ct_fl = (modality == "CT")
            hp = [int(np.sum(preds[:, :, z] > 0)) for z in range(nz_vol)]
            best6 = sorted(np.argsort(hp)[-6:])

            fig, axes = plt.subplots(2, 6, figsize=(24, 8))
            fig.patch.set_facecolor("#1E1E2F")
            for i, z in enumerate(best6):
                sl = raw_vol[:, :, z].astype(np.float32)
                if is_ct_fl:
                    vn_c, vx_c = -200.0, 600.0  # CT_WC=200, CT_WW=800
                    disp = np.clip((sl - vn_c) / (vx_c - vn_c), 0, 1)
                else:
                    p1, p99 = np.percentile(sl, [1, 99])
                    disp = np.clip((sl - p1) / max(p99 - p1, 1e-8), 0, 1)

                axes[0, i].imshow(disp.T, cmap="gray", origin="lower")
                axes[0, i].set_title(f"Slice {z}", color="#ECF0F1", fontsize=10)
                axes[0, i].axis("off")

                ov = (np.stack([disp.T] * 3, -1) * 255).astype(np.uint8)
                for c_id, col in MMWHS_RGB.items():
                    mask = preds[:, :, z].T == c_id
                    ov[mask] = (ov[mask] * 0.35 + np.array(col) * 0.65).astype(np.uint8)
                axes[1, i].imshow(ov, origin="lower")
                axes[1, i].axis("off")

            fig.legend(
                handles=[
                    Patch(facecolor=np.array(MMWHS_RGB[c]) / 255, label=MMWHS_CLASS_NAMES[c])
                    for c in range(1, 8)
                ],
                loc="lower center", ncol=7, fontsize=10,
                facecolor="#2C3E50", edgecolor="#1ABC9C", labelcolor="#ECF0F1",
            )
            fig.suptitle(f"Heart Segmentation — {pid}", fontsize=14, color="#ECF0F1")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.10, top=0.93)
            seg_buf = io.BytesIO()
            plt.savefig(seg_buf, format="png", dpi=120, facecolor="#1E1E2F")
            plt.close(fig)
            seg_buf.seek(0)

        st.markdown("#### 2D Segmentation Preview")
        st.image(
            seg_buf,
            caption="Top 6 slices with most cardiac tissue — raw (top row) and label overlay (bottom row)",
            width="stretch",
        )

        # ── Download prediction NIfTI ─────────────────────────────────────────
        import nibabel as nib_dl
        _inv_lbl = {v: k for k, v in MMWHS_LABEL_MAP.items()}
        pred_out = np.zeros_like(preds, dtype=np.int16)
        for ci, orig_val in _inv_lbl.items():
            pred_out[preds == ci] = orig_val
        pred_nii_obj = nib_dl.Nifti1Image(pred_out, nii.affine, nii.header)
        _pred_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as _tf:
                _pred_tmp = _tf.name
            nib_dl.save(pred_nii_obj, _pred_tmp)
            with open(_pred_tmp, "rb") as _f:
                pred_bytes = _f.read()
        finally:
            if _pred_tmp and os.path.exists(_pred_tmp):
                _safe_unlink(_pred_tmp)

        st.download_button(
            "⬇ Download Prediction NIfTI",
            pred_bytes,
            file_name=f"{pid}_pred.nii.gz",
            mime="application/gzip",
            help="Label values: 205=LV Myo, 420=LV, 500=RV, 550=LA, 600=RA, 820=Aorta, 850=PA",
        )
        st.divider()

        with st.spinner("Extracting 3D meshes…"):
            meshes = {}
            for c in range(1, 8):
                if np.sum(preds == c) < 100:
                    continue
                v, f, n = _extract_smooth_mesh(preds, c, sigma=sigma, step=step,
                                               spacing=spacing)
                if v is not None:
                    meshes[c] = (v, f, n)

        if not meshes:
            st.error("No structures could be meshed. Check that the volume contains cardiac tissue.")
            return

        st.success(
            f"3D meshes ready for {len(meshes)} structure(s): "
            + ", ".join(MMWHS_CLASS_NAMES[c] for c in meshes)
        )

        with st.spinner("Generating interactive viewer…"):
            html = _build_heart_html(meshes, pid)

        st.markdown("#### Interactive 3D Heart Viewer")
        st.caption(
            "Use the panel on the left to toggle structures, explode the view, "
            "adjust opacity and switch rendering mode."
        )
        components.html(html, height=650, scrolling=False)
        st.download_button(
            "Download 3D Viewer (HTML)", html.encode(),
            file_name=f"{pid}_heart_3d.html", mime="text/html",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            _safe_unlink(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2A — ACDC Cardiac MRI Disease Classification
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ACDC cardiac model…")
def _load_acdc():
    import torch
    import torch.nn as nn
    device = _device()
    C = 32

    class RB(nn.Module):
        def __init__(s, ci, co, dp=0.1):
            super().__init__()
            s.c = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            )
            s.sk = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()
            s.d = nn.Dropout2d(dp); s.a = nn.ReLU(True)
        def forward(s, x): return s.a(s.d(s.c(x)) + s.sk(x))

    class AP(nn.Module):
        def __init__(s, ch):
            super().__init__()
            s.a = nn.Sequential(
                nn.Linear(ch, ch // 4), nn.ReLU(True),
                nn.Linear(ch // 4, ch), nn.Sigmoid(),
            )
        def forward(s, x):
            avg = x.mean(dim=(2, 3))
            return (x * s.a(avg)[:, :, None, None]).mean(dim=(2, 3))

    class Net(nn.Module):
        def __init__(s, ns, nc, C):
            super().__init__()
            s.e1 = RB(1, C, .05);   s.e2 = RB(C, C*2, .1)
            s.e3 = RB(C*2, C*4, .1); s.e4 = RB(C*4, C*8, .15)
            s.pool = nn.MaxPool2d(2); s.bot = RB(C*8, C*16, .2)
            s.u4 = nn.ConvTranspose2d(C*16, C*8, 2, stride=2); s.d4 = RB(C*16, C*8, .1)
            s.u3 = nn.ConvTranspose2d(C*8,  C*4, 2, stride=2); s.d3 = RB(C*8,  C*4, .1)
            s.u2 = nn.ConvTranspose2d(C*4,  C*2, 2, stride=2); s.d2 = RB(C*4,  C*2, .05)
            s.u1 = nn.ConvTranspose2d(C*2,  C,   2, stride=2); s.d1 = RB(C*2,  C,   .05)
            s.seg_out = nn.Conv2d(C, ns, 1); s.aux_out = nn.Conv2d(C*4, ns, 1)
            s.ap  = AP(C * 16)
            s.cls = nn.Sequential(
                nn.Linear(C*16, 256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(.4),
                nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(.3), nn.Linear(128, nc),
            )
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1))
            e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
            b  = s.bot(s.pool(e4))
            d4 = s.d4(torch.cat([s.u4(b), e4], 1))
            d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
            d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
            d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return s.seg_out(d1), s.aux_out(d3), s.cls(s.ap(b))

    model = Net(ACDC_NUM_SEG, ACDC_NUM_PATH, C)
    ckpt  = torch.load(ACDC_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval().to(device)
    return model, device, ckpt


def _acdc_infer(model, vol3d: np.ndarray, device: str, tta: bool = True):
    import torch
    import torch.nn.functional as F
    nz = vol3d.shape[2]
    preds = np.zeros(vol3d.shape, dtype=np.int32)
    all_p, all_a = [], []
    for z in range(nz):
        s = _percentile_norm(vol3d[:, :, z])
        t = torch.from_numpy(s).float().unsqueeze(0).unsqueeze(0)
        sa = ca = None; n = 0
        for flip in ([False, True] if tta else [False]):
            ti = torch.flip(t, [-1]) if flip else t.clone()
            tp, _, oh, ow = _pad32(ti, None)
            with torch.no_grad():
                sg, _, cl = model(tp.to(device))
            sp = F.softmax(sg, 1).squeeze(0).cpu().numpy()[:, :oh, :ow]
            if flip:
                sp = sp[:, :, ::-1]
            cp = F.softmax(cl, 1).squeeze(0).cpu().numpy()
            sa = sp if sa is None else sa + sp
            ca = cp if ca is None else ca + cp
            n += 1
        sf = (sa / n).argmax(0)
        preds[:, :, z] = sf[:vol3d.shape[0], :vol3d.shape[1]]
        area = int((preds[:, :, z] > 0).sum())
        if area > 0:
            all_p.append(ca / n); all_a.append(area)
    if all_p:
        w  = np.array(all_a, dtype=float)
        wp = np.average(all_p, axis=0, weights=w / w.sum())
    else:
        wp = np.ones(ACDC_NUM_PATH) / ACDC_NUM_PATH
    return preds, wp


def _render_acdc_tab():
    st.markdown("""
    <div style='background:#273241;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem'>
    <h4 style='color:#1ABC9C;margin:0'>ACDC — Cardiac MRI Disease Classification</h4>
    <p style='color:#BDC3C7;margin:.4rem 0 0'>
    ResBlock U-Net with attention-pooling classification head.<br>
    Classifies 6 cardiac pathologies from short-axis MRI (ED frame required).<br>
    Upload an ES frame as well to compute LV &amp; RV volumes (mL) and ejection fraction.
    </p></div>""", unsafe_allow_html=True)

    ok = os.path.isfile(ACDC_MODEL)
    st.markdown(
        f"<p>{_model_badge('cardiac_v3_best.pth', '✓' if ok else '✗', '#2ECC71' if ok else '#E74C3C')}</p>",
        unsafe_allow_html=True,
    )
    if not ok:
        st.error("Checkpoint not found."); return

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### Upload Frames")
        ed_file = st.file_uploader("ED frame — End Diastole (.nii / .nii.gz)",
                                   type=["nii", "gz"], key="acdc_ed")
        es_file = st.file_uploader("ES frame — End Systole (optional, for EF)",
                                   type=["nii", "gz"], key="acdc_es")
    with col2:
        st.markdown("#### Pathology Classes")
        diag_colors = {0:"#2ECC71",1:"#E74C3C",2:"#3498DB",3:"#F39C12",4:"#9B59B6",5:"#E67E22"}
        for i, name in ACDC_PATH_CLASSES.items():
            st.markdown(
                f"<span style='color:{diag_colors[i]}'>●</span> **{i}.** {name}",
                unsafe_allow_html=True,
            )

    st.divider()
    run = st.button("▶ Run ACDC Diagnosis", type="primary",
                    disabled=ed_file is None, key="acdc_run")
    if not ed_file:
        st.info("Upload an ED NIfTI frame to enable diagnosis."); return
    if not run:
        return

    try:
        import nibabel as nib
        import torch  # noqa: F401
    except ImportError as e:
        st.error(f"Missing dependency: {e}. Run: `pip install nibabel torch`"); return

    tmp_ed = tmp_es = None
    try:
        with st.spinner("Loading model…"):
            model, device, ckpt = _load_acdc()

        with st.spinner("Loading NIfTI volume…"):
            tmp_ed = _save_nifti_tmp(ed_file)
            nii_ed = nib.load(tmp_ed)
            vol_ed = nii_ed.get_fdata(dtype=np.float32)
            if vol_ed.ndim == 4:
                vol_ed = vol_ed[:, :, :, 0]
            zooms = tuple(float(z) for z in nii_ed.header.get_zooms()[:3])

        with st.spinner(f"Running TTA inference on {vol_ed.shape[2]} slices…"):
            preds_ed, probs = _acdc_infer(model, vol_ed, device, tta=True)

        # Clinical Myo/LV ratio heuristic override
        mv = int(np.sum(preds_ed == 2)); lv = int(np.sum(preds_ed == 3))
        if lv > 0:
            r = mv / lv
            if r < 1.0:
                probs[0] += 0.30; probs[3] -= 0.30
            elif r > 1.2:
                probs[3] += 0.30; probs[0] -= 0.30
            probs = np.clip(probs, 1e-6, 1.); probs /= probs.sum()

        diag = int(probs.argmax()); conf = probs[diag] * 100
        st.success(
            f"Done — epoch **{ckpt.get('epoch', '?')}** | "
            f"mDice **{ckpt.get('dice', 0):.4f}** | device **{device.upper()}**"
        )

        # Diagnosis banner
        d_color = diag_colors.get(diag, "#1ABC9C")
        st.markdown(f"""
        <div style='background:{d_color}22;border:2px solid {d_color};
                    border-radius:10px;padding:1rem 1.5rem;margin:.75rem 0'>
        <h3 style='color:{d_color};margin:0'>Diagnosis: {ACDC_PATH_CLASSES[diag]}</h3>
        <p style='color:#ECF0F1;margin:.3rem 0 0'>Confidence: {conf:.1f}%</p>
        </div>""", unsafe_allow_html=True)

        # Probability bar chart
        import altair as alt, pandas as pd
        prob_df = pd.DataFrame([
            {"Pathology": ACDC_PATH_CLASSES[i], "Probability": float(probs[i])}
            for i in range(ACDC_NUM_PATH)
        ])
        chart = (
            alt.Chart(prob_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%", labelColor="#ECF0F1", titleColor="#1ABC9C")),
                y=alt.Y("Pathology:N", sort="-x",
                        axis=alt.Axis(labelColor="#ECF0F1", titleColor="#1ABC9C")),
                color=alt.Color("Probability:Q",
                                scale=alt.Scale(scheme="tealblues"), legend=None),
                tooltip=["Pathology", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=200, background="#273241")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, width="stretch")

        # ── ED volumes in mL (using real voxel spacing from NIfTI header) ──────
        vx, vy, vz   = zooms
        voxel_ml     = (vx * vy * vz) / 1000.0   # mm³ → mL
        rv_ed_ml     = int(np.sum(preds_ed == 1)) * voxel_ml
        myo_ed_ml    = int(np.sum(preds_ed == 2)) * voxel_ml
        lv_edv_ml    = int(np.sum(preds_ed == 3)) * voxel_ml

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RV Volume (ED)", f"{rv_ed_ml:.1f} mL")
        mc2.metric("Myocardium (ED)", f"{myo_ed_ml:.1f} mL")
        mc3.metric("LV EDV", f"{lv_edv_ml:.1f} mL", help="End-Diastolic Volume — normal range 100–150 mL")

        # ── Ejection Fraction panel ───────────────────────────────────────────
        if es_file:
            with st.spinner("Computing ejection fraction…"):
                tmp_es = _save_nifti_tmp(es_file)
                vol_es = nib.load(tmp_es).get_fdata(dtype=np.float32)
                if vol_es.ndim == 4:
                    vol_es = vol_es[:, :, :, 0]
                preds_es, _ = _acdc_infer(model, vol_es, device, tta=False)

                lv_esv_ml   = int(np.sum(preds_es == 3)) * voxel_ml
                rv_esv_ml   = int(np.sum(preds_es == 1)) * voxel_ml

                lv_ef = (lv_edv_ml - lv_esv_ml) / lv_edv_ml * 100 if lv_edv_ml > 0 else 0.0
                rv_ef = (rv_ed_ml  - rv_esv_ml)  / rv_ed_ml  * 100 if rv_ed_ml  > 0 else 0.0

                if lv_ef < 40:
                    ef_col, ef_status, ef_interp, ef_icon = (
                        "#E74C3C", "Reduced EF",
                        "Severely reduced — consistent with DCM / Infarction", "🔴")
                elif lv_ef < 50:
                    ef_col, ef_status, ef_interp, ef_icon = (
                        "#E67E22", "Mildly Reduced",
                        "Mildly reduced EF — borderline systolic dysfunction", "🟠")
                elif lv_ef <= 70:
                    ef_col, ef_status, ef_interp, ef_icon = (
                        "#2ECC71", "Normal",
                        "Normal LV systolic function", "🟢")
                else:
                    ef_col, ef_status, ef_interp, ef_icon = (
                        "#F39C12", "Hyperdynamic",
                        "High EF — possible HCM or dynamic outflow obstruction", "🟡")

                rv_ef_col    = "#E74C3C" if rv_ef < 45 else "#2ECC71"
                rv_ef_status = "Normal" if rv_ef >= 45 else "Reduced"
                ef_bar_pct   = min(lv_ef, 100)

                components.html(f"""
                <div style='background:#1a2535;border-radius:12px;padding:1.4rem 1.6rem;
                            margin:.9rem 0;border:1px solid #2a3f5f;font-family:sans-serif'>

                  <h4 style='color:#1ABC9C;margin:0 0 1.1rem;font-size:1rem;
                              letter-spacing:.03em'>
                    Cardiac Volumes &amp; Ejection Fraction
                  </h4>

                  <!-- LV row -->
                  <div style='color:#8899aa;font-size:.72rem;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:.55rem'>Left Ventricle</div>
                  <div style='display:flex;gap:.8rem;margin-bottom:.6rem'>

                    <div style='flex:1;background:#273241;border-radius:8px;
                                padding:.7rem;text-align:center'>
                      <div style='color:#8899aa;font-size:.72rem;margin-bottom:.2rem'>EDV</div>
                      <div style='color:#ECF0F1;font-size:1.45rem;font-weight:700;
                                  line-height:1'>{lv_edv_ml:.1f}</div>
                      <div style='color:#5D6D7E;font-size:.68rem;margin-top:.2rem'>
                        mL &nbsp;·&nbsp; <span style='color:#4a6070'>norm 100–150</span>
                      </div>
                    </div>

                    <div style='flex:1;background:#273241;border-radius:8px;
                                padding:.7rem;text-align:center'>
                      <div style='color:#8899aa;font-size:.72rem;margin-bottom:.2rem'>ESV</div>
                      <div style='color:#ECF0F1;font-size:1.45rem;font-weight:700;
                                  line-height:1'>{lv_esv_ml:.1f}</div>
                      <div style='color:#5D6D7E;font-size:.68rem;margin-top:.2rem'>
                        mL &nbsp;·&nbsp; <span style='color:#4a6070'>norm 35–60</span>
                      </div>
                    </div>

                    <div style='flex:2;background:{ef_col}18;
                                border:1.5px solid {ef_col}66;
                                border-radius:8px;padding:.7rem;text-align:center'>
                      <div style='color:{ef_col};font-size:.78rem;font-weight:600;
                                  margin-bottom:.1rem'>LVEF {ef_icon}</div>
                      <div style='color:{ef_col};font-size:2.1rem;font-weight:800;
                                  line-height:1'>{lv_ef:.1f}%</div>
                      <div style='color:#BDC3C7;font-size:.72rem;margin-top:.25rem'>
                        {ef_status}
                      </div>
                    </div>

                  </div>

                  <!-- EF progress bar -->
                  <div style='background:#273241;border-radius:6px;height:9px;
                              overflow:hidden;margin-bottom:.18rem'>
                    <div style='width:{ef_bar_pct:.1f}%;background:linear-gradient(
                                  90deg,#E74C3C 0%,#E67E22 38%,#2ECC71 48%,
                                  #2ECC71 70%,#F39C12 82%,#F39C12 100%);
                                height:100%;border-radius:6px'></div>
                  </div>
                  <div style='display:flex;justify-content:space-between;
                              color:#4a6070;font-size:.65rem;margin-bottom:1.1rem'>
                    <span>0%</span>
                    <span style='color:#E74C3C'>40%</span>
                    <span style='color:#2ECC71'>50% – 70%</span>
                    <span style='color:#F39C12'>100%</span>
                  </div>

                  <!-- RV row -->
                  <div style='border-top:1px solid #2a3f5f;padding-top:1rem'>
                    <div style='color:#8899aa;font-size:.72rem;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:.55rem'>Right Ventricle</div>
                    <div style='display:flex;gap:.8rem'>

                      <div style='flex:1;background:#273241;border-radius:8px;
                                  padding:.7rem;text-align:center'>
                        <div style='color:#8899aa;font-size:.72rem;margin-bottom:.2rem'>EDV</div>
                        <div style='color:#ECF0F1;font-size:1.45rem;font-weight:700;
                                    line-height:1'>{rv_ed_ml:.1f}</div>
                        <div style='color:#5D6D7E;font-size:.68rem;margin-top:.2rem'>mL</div>
                      </div>

                      <div style='flex:1;background:#273241;border-radius:8px;
                                  padding:.7rem;text-align:center'>
                        <div style='color:#8899aa;font-size:.72rem;margin-bottom:.2rem'>ESV</div>
                        <div style='color:#ECF0F1;font-size:1.45rem;font-weight:700;
                                    line-height:1'>{rv_esv_ml:.1f}</div>
                        <div style='color:#5D6D7E;font-size:.68rem;margin-top:.2rem'>mL</div>
                      </div>

                      <div style='flex:2;background:{rv_ef_col}18;
                                  border:1.5px solid {rv_ef_col}66;
                                  border-radius:8px;padding:.7rem;text-align:center'>
                        <div style='color:{rv_ef_col};font-size:.78rem;font-weight:600;
                                    margin-bottom:.1rem'>RVEF</div>
                        <div style='color:{rv_ef_col};font-size:2.1rem;font-weight:800;
                                    line-height:1'>{rv_ef:.1f}%</div>
                        <div style='color:#BDC3C7;font-size:.72rem;margin-top:.25rem'>
                          {rv_ef_status}
                          <span style='color:#4a6070'> · norm ≥45%</span>
                        </div>
                      </div>

                    </div>
                  </div>

                  <!-- Clinical note -->
                  <div style='background:{ef_col}15;border-left:3px solid {ef_col};
                              border-radius:0 6px 6px 0;padding:.6rem .9rem;
                              margin-top:1rem'>
                    <span style='color:{ef_col};font-weight:600'>Clinical Note: </span>
                    <span style='color:#BDC3C7'>{ef_interp}</span>
                  </div>

                </div>""", height=480)

        # Slice visualisation
        with st.spinner("Rendering slice preview…"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            SEG_COLORS = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
            nz = vol_ed.shape[2]
            hp = [int(np.sum(preds_ed[:, :, z] > 0)) for z in range(nz)]
            bz = sorted(np.argsort(hp)[-min(6, nz):])
            ns = len(bz)
            fig, axes = plt.subplots(2, ns, figsize=(4 * ns, 8), facecolor="#0d0d0d")
            if ns == 1:
                axes = axes[:, np.newaxis]
            for i, z in enumerate(bz):
                s = _percentile_norm(vol_ed[:, :, z])
                axes[0, i].imshow(s.T, cmap="gray", origin="lower")
                axes[0, i].set_title(f"Slice {z}", color="#eee", fontsize=9)
                axes[0, i].axis("off")
                ov = (np.stack([s.T] * 3, -1) * 255).astype(np.uint8)
                for c, col in SEG_COLORS.items():
                    m = preds_ed[:, :, z].T == c
                    ov[m] = (ov[m] * .4 + np.array(col) * .6).astype(np.uint8)
                axes[1, i].imshow(ov, origin="lower")
                axes[1, i].axis("off")
            legend = [Patch(facecolor=np.array(SEG_COLORS[c]) / 255,
                            label=ACDC_SEG_CLASSES[c]) for c in range(1, ACDC_NUM_SEG)]
            fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
                       facecolor="#111", labelcolor="white")
            fig.suptitle(
                f"Cardiac MRI — {ed_file.name}\n{ACDC_PATH_CLASSES[diag]} ({conf:.1f}%)",
                fontsize=13, color="white",
            )
            plt.tight_layout(); plt.subplots_adjust(bottom=.08, top=.90)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=130, facecolor="#0d0d0d")
            plt.close(); buf.seek(0)

        st.markdown("#### Slice Segmentation Preview")
        st.image(buf, width="stretch")
        st.download_button(
            "Download Visualization", buf.getvalue(),
            file_name=f"acdc_{ed_file.name.split('.')[0]}_result.png", mime="image/png",
        )

        # Optional 3D problem analysis
        with st.expander("3D Problem Analysis (optional — may be slow)"):
            if st.button("Run 3D Analysis", key="acdc_3d_btn"):
                with st.spinner("Detecting segmentation problems…"):
                    from scipy import ndimage
                    from scipy.ndimage import binary_dilation
                    sx, sy, sz = zooms
                    problems = {k: [] for k in ["fragment", "hole", "thin_wall", "overlap", "jump"]}
                    report = []
                    Z = preds_ed.shape[2]

                    for c in range(1, ACDC_NUM_SEG):
                        mask = (preds_ed == c).astype(np.uint8)
                        labeled, n_total = ndimage.label(mask)
                        if n_total > 1:
                            sizes = ndimage.sum(mask, labeled, range(1, n_total + 1))
                            main  = np.argmax(sizes) + 1
                            for lbl in range(1, n_total + 1):
                                if lbl == main: continue
                                frag = np.argwhere(labeled == lbl)
                                if len(frag) < 5: continue
                                problems["fragment"].extend(frag.tolist())
                                s_ = sorted(set(frag[:, 2].tolist()))
                                report.append(
                                    f"[FRAGMENT] {ACDC_SEG_CLASSES[c]}: {len(frag)} voxels, "
                                    f"slices {s_[0]}–{s_[-1]}"
                                )

                    for c in [2, 3]:
                        mask   = (preds_ed == c).astype(np.uint8)
                        filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
                        hm     = (filled - mask) > 0
                        if hm.any():
                            lh, nh = ndimage.label(hm)
                            for lbl in range(1, nh + 1):
                                hv = np.argwhere(lh == lbl)
                                if len(hv) < 3: continue
                                problems["hole"].extend(hv.tolist())
                                s_ = sorted(set(hv[:, 2].tolist()))
                                report.append(
                                    f"[HOLE] {ACDC_SEG_CLASSES[c]}: {len(hv)} voxels, "
                                    f"slices {s_[0]}–{s_[-1]}"
                                )

                    rv_d = binary_dilation((preds_ed == 1), iterations=2)
                    lv_d = binary_dilation((preds_ed == 3), iterations=2)
                    susp = rv_d & lv_d & ~(preds_ed == 2) & ((preds_ed == 1) | (preds_ed == 3))
                    if susp.any():
                        ov = np.argwhere(susp)
                        problems["overlap"] = ov.tolist()
                        report.append(f"[OVERLAP] RV/LV confusion in {len(ov)} voxels")

                    total = sum(len(v) for v in problems.values())
                    if total == 0:
                        st.success("No segmentation problems detected.")
                    else:
                        st.warning(f"{total:,} problematic voxels detected.")
                        for line in report:
                            st.markdown(f"- {line}")
                        try:
                            import plotly.graph_objects as go
                            problem_colors = {
                                "fragment": "orange", "hole": "magenta",
                                "thin_wall": "yellow", "overlap": "red", "jump": "cyan",
                            }
                            fig3 = go.Figure()
                            for key, colour in problem_colors.items():
                                coords = np.array(problems[key]) if problems[key] else None
                                if coords is None or len(coords) == 0: continue
                                if len(coords) > 2000:
                                    coords = coords[np.random.choice(len(coords), 2000, replace=False)]
                                fig3.add_trace(go.Scatter3d(
                                    x=coords[:, 0] * sx, y=coords[:, 1] * sy, z=coords[:, 2] * sz,
                                    mode="markers",
                                    marker=dict(size=3, color=colour, opacity=0.8),
                                    name=key.replace("_", " ").title(),
                                ))
                            fig3.update_layout(
                                paper_bgcolor="#0d0d0d",
                                scene=dict(bgcolor="#111",
                                           xaxis=dict(color="#888", title="X mm"),
                                           yaxis=dict(color="#888", title="Y mm"),
                                           zaxis=dict(color="#888", title="Z mm")),
                                height=500, margin=dict(l=0, r=0, t=30, b=0),
                            )
                            st.plotly_chart(fig3, width="stretch")
                        except ImportError:
                            pass

        st.markdown("""
        <div style='background:#1E1E2F;border-radius:8px;padding:.75rem 1rem;margin-top:1rem;
                    border-left:4px solid #E67E22'>
        <span style='color:#E67E22;font-weight:700'>⚕ Clinical Disclaimer</span><br>
        <span style='color:#BDC3C7;font-size:.875rem'>Research tool only. Not validated for
        clinical use. All decisions must be made by qualified medical professionals.</span>
        </div>""", unsafe_allow_html=True)

    finally:
        for tp in [tmp_ed, tmp_es]:
            if tp and os.path.exists(tp):
                _safe_unlink(tp)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2B — MyoPS Myocardial Scar & Edema Detection
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading MyoPS model…")
def _load_myops():
    import torch
    import torch.nn as nn
    device = _device()

    class ResBlock(nn.Module):
        def __init__(s, ci, co, dp=0.1):
            super().__init__()
            s.c = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            )
            s.sk   = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()
            s.drop = nn.Dropout2d(dp); s.act = nn.ReLU(True)
        def forward(s, x): return s.act(s.drop(s.c(x)) + s.sk(x))

    class ASPP(nn.Module):
        def __init__(s, ch):
            super().__init__()
            s.b0 = nn.Conv2d(ch, ch // 4, 1, bias=False)
            s.b1 = nn.Conv2d(ch, ch // 4, 3, padding=6,  dilation=6,  bias=False)
            s.b2 = nn.Conv2d(ch, ch // 4, 3, padding=12, dilation=12, bias=False)
            s.b3 = nn.Conv2d(ch, ch // 4, 3, padding=18, dilation=18, bias=False)
            s.proj = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False),
                                   nn.BatchNorm2d(ch), nn.ReLU(True))
        def forward(s, x):
            return s.proj(torch.cat([s.b0(x), s.b1(x), s.b2(x), s.b3(x)], 1))

    class MyoPSNet(nn.Module):
        def __init__(s, ns=6, ch=32):
            super().__init__()
            s.e1 = ResBlock(3, ch, .05); s.e2 = ResBlock(ch, ch*2, .10)
            s.e3 = ResBlock(ch*2, ch*4, .10); s.e4 = ResBlock(ch*4, ch*8, .15)
            s.pool     = nn.MaxPool2d(2)
            s.bot_pre  = ResBlock(ch*8, ch*16, .20)
            s.aspp     = ASPP(ch * 16)
            s.bot_post = ResBlock(ch*16, ch*16, .15)
            s.u4 = nn.ConvTranspose2d(ch*16, ch*8, 2, stride=2); s.d4 = ResBlock(ch*16, ch*8, .10)
            s.u3 = nn.ConvTranspose2d(ch*8,  ch*4, 2, stride=2); s.d3 = ResBlock(ch*8,  ch*4, .10)
            s.u2 = nn.ConvTranspose2d(ch*4,  ch*2, 2, stride=2); s.d2 = ResBlock(ch*4,  ch*2, .05)
            s.u1 = nn.ConvTranspose2d(ch*2,  ch,   2, stride=2); s.d1 = ResBlock(ch*2,  ch,   .05)
            s.seg_out   = nn.Conv2d(ch, ns, 1)
            s.aux_out   = nn.Conv2d(ch * 4, ns, 1)
            s.path_conv = nn.Sequential(ResBlock(ch*2, ch, .05), nn.Conv2d(ch, 2, 1))
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1))
            e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
            b  = s.bot_post(s.aspp(s.bot_pre(s.pool(e4))))
            d4 = s.d4(torch.cat([s.u4(b), e4], 1))
            d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
            d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
            d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return s.seg_out(d1), s.aux_out(d3), s.path_conv(d2)

    model = MyoPSNet(ns=MYOPS_NUM_SEG, ch=32).to(device)
    ckpt  = torch.load(MYOPS_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, device, ckpt


def _myops_infer(model, c0: np.ndarray, de: np.ndarray, t2: np.ndarray,
                 device: str, size: int = 256) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    from scipy.ndimage import binary_dilation

    H, W, nz = c0.shape
    preds = np.zeros((H, W, nz), dtype=np.int32)

    for z in range(nz):
        img = np.stack([_percentile_norm(c0[:, :, z]),
                        _percentile_norm(de[:, :, z]),
                        _percentile_norm(t2[:, :, z])], 0)
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)

        sa = None; n = 0
        for flip in [False, True]:
            for rot_k in [0, 1, 2, 3]:
                ti = torch.flip(t, [-1]) if flip else t.clone()
                ti = torch.rot90(ti, rot_k, [-2, -1])
                tp, _, oh, ow = _pad32(ti, None)
                with torch.no_grad():
                    sg, _, _ = model(tp.to(device))
                sp = F.softmax(sg, 1).squeeze(0).cpu()[:, :oh, :ow]
                sp = torch.rot90(sp, -rot_k, [-2, -1])
                if flip:
                    sp = torch.flip(sp, [-1])
                sa = sp if sa is None else sa + sp
                n += 1

        sa_full = F.interpolate(
            (sa / n).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        preds[:, :, z] = sa_full.argmax(0).numpy()

    # Post-processing: keep largest component per pathology class per slice
    for z in range(nz):
        for cls in [4, 5]:
            mask = preds[:, :, z] == cls
            if not mask.any(): continue
            labeled, n_comp = ndimage.label(mask)
            if n_comp <= 1: continue
            sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
            keep  = np.argmax(sizes) + 1
            preds[:, :, z][mask & (labeled != keep)] = 3

    # Enforce anatomy: scar/edema must be within dilated myocardium
    for z in range(nz):
        myo_region = binary_dilation(preds[:, :, z] == 3, iterations=4)
        preds[:, :, z][(preds[:, :, z] == 4) & ~myo_region] = 3
        preds[:, :, z][(preds[:, :, z] == 5) & ~myo_region] = 0

    return preds


def _myops_analyse_regions(preds: np.ndarray, zooms) -> list:
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt

    AHA = {
        0: "Anterior", 1: "Anteroseptal", 2: "Inferoseptal",
        3: "Inferior", 4: "Inferolateral", 5: "Anterolateral",
    }
    sx, sy, sz = zooms
    vvol = sx * sy * sz
    lv_mask = (preds == 1)
    if lv_mask.any():
        com = ndimage.center_of_mass(lv_mask)
        lv_cx, lv_cy = com[0], com[1]
    else:
        lv_cx, lv_cy = preds.shape[0] // 2, preds.shape[1] // 2

    regions = []
    for cls_id, cls_name in [(4, "Scar"), (5, "Edema")]:
        mask_3d = (preds == cls_id)
        if not mask_3d.any(): continue
        labeled_3d, n_comp = ndimage.label(mask_3d)
        for lbl_id in range(1, n_comp + 1):
            rm = labeled_3d == lbl_id
            if rm.sum() < 10: continue
            com = ndimage.center_of_mass(rm)
            cx, cy, cz = com
            slices_p = sorted(set(np.where(rm.any(axis=(0, 1)))[0].tolist()))
            angle   = np.degrees(np.arctan2(cy - lv_cy, cx - lv_cx)) % 360
            segment = AHA[int(angle // 60)]
            myo_2d  = (preds[:, :, int(cz)] == 3)
            if myo_2d.any():
                dist_lv   = distance_transform_edt(~(preds[:, :, int(cz)] == 1), sampling=(sx, sy))
                myo_thick = float(dist_lv[myo_2d].max()) if len(dist_lv[myo_2d]) else 0
                reg_dist  = float(dist_lv[int(cx), int(cy)])
                transmurality = min(reg_dist / max(myo_thick, 1e-8), 1.0)
            else:
                transmurality = 0.0
            regions.append({
                "class": cls_name, "class_id": cls_id, "voxels": int(rm.sum()),
                "volume_mm3": int(rm.sum()) * vvol,
                "centroid": (cx, cy, cz), "slices": slices_p,
                "wall_segment": segment, "transmurality": transmurality,
                "region_id": lbl_id,
            })
    return regions


def _render_myops_tab():
    st.markdown("""
    <div style='background:#273241;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem'>
    <h4 style='color:#1ABC9C;margin:0'>MyoPS 2020 — Myocardial Scar &amp; Edema Detection</h4>
    <p style='color:#BDC3C7;margin:.4rem 0 0'>
    Multi-sequence U-Net (C0 / DE / T2) with ASPP bottleneck and 8-fold TTA.<br>
    Detects scar and edema, localises them by AHA wall segment and estimates transmurality.
    </p></div>""", unsafe_allow_html=True)

    ok = os.path.isfile(MYOPS_MODEL)
    st.markdown(
        f"<p>{_model_badge('myops_best.pth', '✓' if ok else '✗', '#2ECC71' if ok else '#E74C3C')}</p>",
        unsafe_allow_html=True,
    )
    if not ok:
        st.error("Checkpoint not found."); return

    st.markdown("#### Upload MRI Sequences")
    c1, c2, c3 = st.columns(3)
    with c1:
        c0_file = st.file_uploader("C0 — bSSFP", type=["nii", "gz"], key="myops_c0")
    with c2:
        de_file = st.file_uploader("DE — Late Gadolinium (LGE)", type=["nii", "gz"], key="myops_de")
    with c3:
        t2_file = st.file_uploader("T2 — T2-weighted", type=["nii", "gz"], key="myops_t2")

    all_up = c0_file and de_file and t2_file
    run = st.button("▶ Run MyoPS Analysis", type="primary",
                    disabled=not all_up, key="myops_run")

    if not all_up:
        st.info("Upload all three MRI sequences (C0, DE, T2) to enable analysis."); return
    if not run:
        return

    try:
        import nibabel as nib
        import torch  # noqa: F401
    except ImportError as e:
        st.error(f"Missing: {e}. "
                 "Run: `pip install nibabel torch torchvision scikit-image scipy`"); return

    tmp_c0 = tmp_de = tmp_t2 = None
    try:
        with st.spinner("Loading model…"):
            model, device, ckpt = _load_myops()
            ep   = ckpt.get("epoch", "?")
            md   = ckpt.get("dice",  0)
            sc_d = ckpt.get("scar_dice",  0)
            ed_d = ckpt.get("edema_dice", 0)

        with st.spinner("Loading NIfTI volumes…"):
            tmp_c0 = _save_nifti_tmp(c0_file)
            tmp_de = _save_nifti_tmp(de_file)
            tmp_t2 = _save_nifti_tmp(t2_file)
            nii_c0 = nib.load(tmp_c0)
            zooms  = tuple(float(z) for z in nii_c0.header.get_zooms()[:3])
            c0_vol = nii_c0.get_fdata(dtype=np.float32)
            de_vol = nib.load(tmp_de).get_fdata(dtype=np.float32)
            t2_vol = nib.load(tmp_t2).get_fdata(dtype=np.float32)
            for arr in [c0_vol, de_vol, t2_vol]:
                if arr.ndim == 4:
                    arr = arr[:, :, :, 0]
            if c0_vol.ndim == 4: c0_vol = c0_vol[:, :, :, 0]
            if de_vol.ndim == 4: de_vol = de_vol[:, :, :, 0]
            if t2_vol.ndim == 4: t2_vol = t2_vol[:, :, :, 0]

        st.info(
            f"Volume: **{c0_vol.shape}** | "
            f"Voxel: {zooms[0]:.2f}×{zooms[1]:.2f}×{zooms[2]:.2f} mm | "
            f"Epoch **{ep}** | mDice **{md:.4f}** | scar **{sc_d:.4f}** | edema **{ed_d:.4f}**"
        )

        with st.spinner(f"Running 8-fold TTA inference on {c0_vol.shape[2]} slices…"):
            preds = _myops_infer(model, c0_vol, de_vol, t2_vol, device)

        st.success(f"Inference complete | device **{device.upper()}**")

        # Summary metrics
        sx, sy, sz = zooms; vvol = sx * sy * sz
        scar_vox  = int(np.sum(preds == 4)); edema_vox = int(np.sum(preds == 5))
        myo_vox   = int(np.sum(preds == 3))
        scar_mm3  = scar_vox * vvol; edema_mm3 = edema_vox * vvol; myo_mm3 = myo_vox * vvol
        scar_pct  = scar_mm3  / max(myo_mm3, 1) * 100
        edema_pct = edema_mm3 / max(myo_mm3, 1) * 100
        scar_ed_ratio = scar_vox / max(edema_vox, 1)

        ms1, ms2, ms3, ms4 = st.columns(4)
        ms1.metric("Scar volume",   f"{scar_mm3:.0f} mm³",   f"{scar_pct:.1f}% Myo")
        ms2.metric("Edema volume",  f"{edema_mm3:.0f} mm³",  f"{edema_pct:.1f}% Myo")
        ms3.metric("Myocardium",    f"{myo_mm3:.0f} mm³")
        ms4.metric("Scar/Edema",    f"{scar_ed_ratio:.2f}",  help=">0.7 = mostly scarred")

        # Clinical interpretation
        if scar_mm3 > 0:
            burden = scar_pct
            if burden < 5:
                interp = "Small/focal infarct — likely preserved function"
            elif burden < 15:
                interp = "Moderate infarct — possible regional dysfunction"
            else:
                interp = "Extensive infarct — significant dysfunction likely"
            phase = ("Acute/subacute phase — salvageable myocardium present"
                     if edema_mm3 > scar_mm3
                     else "Chronic infarct or healed injury pattern"
                     if scar_mm3 > 0 and edema_mm3 < scar_mm3 * 0.5
                     else "")
            scar_color = "#E74C3C" if burden > 15 else "#F39C12" if burden > 5 else "#2ECC71"
            st.markdown(f"""
            <div style='background:{scar_color}22;border:1px solid {scar_color};
                        border-radius:8px;padding:.75rem 1rem;margin:.75rem 0;
                        border-left:4px solid {scar_color}'>
            <b style='color:{scar_color}'>{interp}</b><br>
            <span style='color:#BDC3C7'>{phase}</span>
            </div>""", unsafe_allow_html=True)

        # Region analysis
        with st.spinner("Analysing pathology regions…"):
            regions = _myops_analyse_regions(preds, zooms)

        # 3D Plotly viewer
        with st.spinner("Building 3D viewer…"):
            try:
                import plotly.graph_objects as go
                from skimage.measure import marching_cubes
                fig3 = go.Figure()

                for cls_id in [3, 2, 1, 5, 4]:
                    mask = (preds == cls_id)
                    if not mask.any(): continue
                    r, g, b = MYOPS_COLOURS.get(cls_id, (200, 200, 200))
                    try:
                        verts, faces, _, _ = marching_cubes(
                            mask.astype(float), level=0.5, step_size=2, allow_degenerate=False)
                        verts_mm = verts * np.array(zooms)
                        fig3.add_trace(go.Mesh3d(
                            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color=f"rgb({r},{g},{b})",
                            opacity=MYOPS_OPACITY.get(cls_id, 0.5),
                            name=MYOPS_CLASSES[cls_id], showlegend=True,
                            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
                            lightposition=dict(x=100, y=200, z=150),
                        ))
                    except Exception:
                        pass

                # Region centroid markers
                for cls_id, cls_name, colour in [(4, "Scar", "gold"), (5, "Edema", "cyan")]:
                    cls_r = [r for r in regions if r["class_id"] == cls_id]
                    if not cls_r: continue
                    fig3.add_trace(go.Scatter3d(
                        x=[r["centroid"][0] * sx for r in cls_r],
                        y=[r["centroid"][1] * sy for r in cls_r],
                        z=[r["centroid"][2] * sz for r in cls_r],
                        mode="markers+text",
                        marker=dict(size=10, color=colour, symbol="diamond",
                                    line=dict(color="white", width=1.5)),
                        text=[f" {r['class']}<br> {r['volume_mm3']:.0f} mm³" for r in cls_r],
                        textfont=dict(color=colour, size=10),
                        textposition="middle right",
                        hovertext=[
                            f"{r['class']} R{r['region_id']}<br>"
                            f"Vol: {r['volume_mm3']:.1f} mm³<br>"
                            f"Wall: {r['wall_segment']}<br>"
                            f"Transmurality: {r['transmurality']*100:.0f}%<br>"
                            f"Slices: {r['slices'][0]}–{r['slices'][-1]}"
                            for r in cls_r
                        ],
                        hoverinfo="text",
                        name=f"{cls_name} centroids", showlegend=True,
                    ))

                pid_str = c0_file.name.split(".")[0]
                fig3.update_layout(
                    paper_bgcolor="#080810",
                    scene=dict(
                        bgcolor="#0d0d1a",
                        xaxis=dict(title="X mm", color="#888",
                                   backgroundcolor="#0d0d1a", gridcolor="#222", showbackground=True),
                        yaxis=dict(title="Y mm", color="#888",
                                   backgroundcolor="#0d0d1a", gridcolor="#222", showbackground=True),
                        zaxis=dict(title="Slice mm", color="#888",
                                   backgroundcolor="#0d0d1a", gridcolor="#222", showbackground=True),
                        aspectmode="data",
                        camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
                    ),
                    legend=dict(x=0.01, y=0.99,
                                bgcolor="rgba(10,10,20,.88)", bordercolor="#333",
                                font=dict(color="#ddd", size=11)),
                    margin=dict(l=0, r=0, t=50, b=0), height=620,
                    title=dict(
                        text=(f"MyoPS 3D — {pid_str}<br>"
                              f"<span style='font-size:12px;color:#aaa'>"
                              f"Scar: {scar_mm3:.0f} mm³ ({scar_pct:.1f}%) | "
                              f"Edema: {edema_mm3:.0f} mm³ ({edema_pct:.1f}%)</span>"),
                        x=0.02, font=dict(size=15, color="#eee"),
                    ),
                )
                st.markdown("#### Interactive 3D Viewer")
                st.plotly_chart(fig3, width="stretch")
            except ImportError as e:
                st.warning(f"Plotly unavailable: {e}")
            except Exception as e:
                st.warning(f"3D viewer error: {e}")

        # Slice panels
        with st.spinner("Rendering slice panels…"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import Rectangle

            SEG_COLORS_RGB = {
                1: (0.39, 0.58, 0.93), 2: (0.27, 0.51, 0.71),
                3: (0.82, 0.31, 0.31), 4: (1.00, 0.84, 0.00), 5: (0.00, 0.86, 0.86),
            }
            path_slices = sorted(set(
                s for r in regions for s in r["slices"] if r["class_id"] in (4, 5)
            ))
            if not path_slices:
                hp = [int(np.sum(preds[:, :, z] > 0)) for z in range(preds.shape[2])]
                path_slices = sorted(np.argsort(hp)[-4:])
            show_slices = path_slices[:min(6, len(path_slices))]
            n_rows = len(show_slices)

            if n_rows > 0:
                fig_s, axes = plt.subplots(
                    n_rows, 4, figsize=(20, 4.5 * n_rows), facecolor="#0a0a0a"
                )
                if n_rows == 1:
                    axes = axes[np.newaxis, :]

                for row, z in enumerate(show_slices):
                    c0s = _percentile_norm(c0_vol[:, :, z])
                    des = _percentile_norm(de_vol[:, :, z])
                    t2s = _percentile_norm(t2_vol[:, :, z])
                    for col, (img, ttl) in enumerate([
                        (c0s, "C0 (bSSFP)"), (des, "DE (LGE)"), (t2s, "T2")
                    ]):
                        ax = axes[row, col]
                        ax.imshow(img.T, cmap="gray", origin="lower")
                        ax.set_title(ttl, fontsize=8, color="#ccc", pad=3)
                        ax.axis("off"); ax.set_facecolor("#0a0a0a")

                    ax_ann = axes[row, 3]
                    base = np.stack([des.T] * 3, -1).copy()
                    for c, rgb in SEG_COLORS_RGB.items():
                        m = preds[:, :, z].T == c
                        if m.any():
                            base[m] = base[m] * 0.3 + np.array(rgb) * 0.7
                    ax_ann.imshow(np.clip(base, 0, 1), origin="lower")
                    ax_ann.set_title(f"Segmentation z={z}", fontsize=8, color="white", pad=3)
                    ax_ann.axis("off"); ax_ann.set_facecolor("#0a0a0a")

                    for r in regions:
                        if z not in r["slices"]: continue
                        cls_mask = (preds[:, :, z] == r["class_id"])
                        if not cls_mask.any(): continue
                        ys_c, xs_c = np.where(cls_mask.T)
                        if len(xs_c) == 0: continue
                        x0, x1 = xs_c.min(), xs_c.max()
                        y0, y1 = ys_c.min(), ys_c.max()
                        colour = "gold" if r["class_id"] == 4 else "cyan"
                        rect = Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4,
                                         linewidth=1.5, edgecolor=colour,
                                         facecolor="none", linestyle="--")
                        ax_ann.add_patch(rect)
                        ax_ann.text(
                            x0, y0 - 4,
                            f"{r['class']}\n{r['volume_mm3']:.0f} mm³\n{r['wall_segment']}",
                            color=colour, fontsize=6.5, fontweight="bold", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a0a0a",
                                      alpha=0.75, edgecolor="none"),
                        )
                        cx, cy, _ = r["centroid"]
                        ax_ann.plot(cy, cx, "o", color=colour, markersize=5,
                                    markeredgecolor="white", markeredgewidth=0.8)

                legend_items = [
                    mpatches.Patch(color=SEG_COLORS_RGB[c], label=MYOPS_CLASSES[c])
                    for c in range(1, MYOPS_NUM_SEG)
                ]
                fig_s.legend(handles=legend_items, loc="lower center", ncol=5,
                             fontsize=9, facecolor="#111", edgecolor="#444",
                             labelcolor="white", framealpha=0.9)
                fig_s.suptitle(
                    f"MyoPS Pathology Slices — {c0_file.name.split('.')[0]}\n"
                    f"Scar: {scar_vox:,} vox  |  Edema: {edema_vox:,} vox  |  "
                    f"Gold = Scar · Cyan = Edema",
                    fontsize=11, fontweight="bold", color="white", y=0.995,
                )
                plt.tight_layout(rect=[0, 0.04, 1, 0.985])
                buf = io.BytesIO()
                fig_s.savefig(buf, format="png", dpi=120,
                              facecolor="#0a0a0a", bbox_inches="tight")
                plt.close(); buf.seek(0)
                st.markdown("#### Slice Panels")
                st.image(buf, width="stretch")
                st.download_button(
                    "Download Slice Panels", buf.getvalue(),
                    file_name=f"myops_{c0_file.name.split('.')[0]}_slices.png",
                    mime="image/png",
                )

        # Text report
        with st.expander("Clinical Report"):
            report_lines = [
                "=" * 62, "  MyoPS PATHOLOGY REPORT", "=" * 62,
                f"  Patient:   {c0_file.name.split('.')[0]}",
                f"  Model:     myops_best.pth  (epoch {ep})",
                f"  Voxel:     {zooms[0]:.2f} × {zooms[1]:.2f} × {zooms[2]:.2f} mm",
                "", "  VOLUMES", "  " + "-" * 40,
                f"  Myocardium     {myo_vox:>8,} vox  {myo_mm3:>9.1f} mm3",
                f"  Scar           {scar_vox:>8,} vox  {scar_mm3:>9.1f} mm3",
                f"  Edema          {edema_vox:>8,} vox  {edema_mm3:>9.1f} mm3",
                "", "  PATHOLOGY INDICES", "  " + "-" * 40,
                f"  Scar burden    {scar_pct:>6.1f}%  of myocardium",
                f"  Edema extent   {edema_pct:>6.1f}%  of myocardium",
                f"  Scar/Edema     {scar_ed_ratio:>6.2f}   (>0.7 = mostly scarred)",
                "", "  DETECTED REGIONS", "  " + "-" * 40,
            ]
            for r in regions:
                report_lines.append(
                    f"  [{r['class']:<5} R{r['region_id']}]  "
                    f"{r['volume_mm3']:>7.1f} mm3  "
                    f"wall={r['wall_segment']:<15}  "
                    f"{r['transmurality'] * 100:.0f}% transmural  "
                    f"slices {r['slices'][0]}–{r['slices'][-1]}"
                )
            scar_walls = sorted(set(r["wall_segment"] for r in regions if r["class_id"] == 4))
            if scar_walls:
                report_lines.append(f"  Affected wall segments: {', '.join(scar_walls)}")
            report_lines.append("=" * 62)
            report_text = "\n".join(report_lines)
            st.code(report_text, language=None)
            st.download_button(
                "Download Report (TXT)", report_text.encode(),
                file_name=f"myops_{c0_file.name.split('.')[0]}_report.txt",
                mime="text/plain",
            )

        st.markdown("""
        <div style='background:#1E1E2F;border-radius:8px;padding:.75rem 1rem;margin-top:1rem;
                    border-left:4px solid #E67E22'>
        <span style='color:#E67E22;font-weight:700'>⚕ Clinical Disclaimer</span><br>
        <span style='color:#BDC3C7;font-size:.875rem'>Research tool only. Not validated for
        clinical use. All decisions must be made by qualified medical professionals.</span>
        </div>""", unsafe_allow_html=True)

    finally:
        for tp in [tmp_c0, tmp_de, tmp_t2]:
            if tp and os.path.exists(tp):
                _safe_unlink(tp)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2C — LASC 2018  Left Atrial Segmentation
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading LASC Left-Atrial model…")
def _load_lasc():
    import torch
    import torch.nn as nn

    device = _device()

    class ResBlock(nn.Module):
        # Attribute names MUST match the checkpoint: s.c (conv path), s.sk (skip)
        def __init__(s, ic, oc, drop=0.1):
            super().__init__()
            s.c = nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.Dropout2d(drop),
            )
            s.sk = nn.Conv2d(ic, oc, 1, bias=False) if ic != oc else nn.Identity()
        def forward(s, x): return torch.relu(s.c(x) + s.sk(x))

    class ASPP(nn.Module):
        # bias=False on all branches — matches how the checkpoint was trained
        def __init__(s, ch):
            super().__init__()
            q = ch // 4
            s.b0 = nn.Conv2d(ch, q, 1,  bias=False)
            s.b1 = nn.Conv2d(ch, q, 3,  padding=6,  dilation=6,  bias=False)
            s.b2 = nn.Conv2d(ch, q, 3,  padding=12, dilation=12, bias=False)
            s.b3 = nn.Conv2d(ch, q, 3,  padding=18, dilation=18, bias=False)
            s.proj = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True))
        def forward(s, x): return s.proj(torch.cat([s.b0(x), s.b1(x), s.b2(x), s.b3(x)], 1))

    class LASCNet(nn.Module):
        def __init__(s, n_cls=3, ch=32):
            super().__init__()
            s.e1 = ResBlock(1,    ch);    s.e2 = ResBlock(ch,   ch*2)
            s.e3 = ResBlock(ch*2, ch*4); s.e4 = ResBlock(ch*4, ch*8)
            s.pool = nn.MaxPool2d(2)
            s.bot_pre  = ResBlock(ch*8,  ch*16)
            s.aspp     = ASPP(ch*16)
            s.bot_post = ResBlock(ch*16, ch*16)
            s.u4 = nn.ConvTranspose2d(ch*16, ch*8,  2, stride=2); s.d4 = ResBlock(ch*16, ch*8)
            s.u3 = nn.ConvTranspose2d(ch*8,  ch*4,  2, stride=2); s.d3 = ResBlock(ch*8,  ch*4)
            s.u2 = nn.ConvTranspose2d(ch*4,  ch*2,  2, stride=2); s.d2 = ResBlock(ch*4,  ch*2)
            s.u1 = nn.ConvTranspose2d(ch*2,  ch,    2, stride=2); s.d1 = ResBlock(ch*2,  ch)
            s.seg_out = nn.Conv2d(ch, n_cls, 1)
            s.aux_out = nn.Conv2d(ch*4, n_cls, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1))
            e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
            b  = s.bot_post(s.aspp(s.bot_pre(s.pool(e4))))
            d4 = s.d4(torch.cat([s.u4(b),  e4], 1))
            d3 = s.d3(torch.cat([s.u3(d4), e3], 1))
            d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
            d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            return s.seg_out(d1), s.aux_out(d3)

    model = LASCNet(LASC_NUM_CLASSES, ch=32)
    ckpt  = torch.load(LASC_MODEL, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, device


def _lasc_load_volume(path: str):
    """Load a 3D LGE-MRI volume from .nrrd or .nii/.nii.gz.
    Returns (volume_ndarray, voxel_spacing_mm_tuple)."""
    ext = path.lower()
    if ext.endswith(".nrrd"):
        try:
            import nrrd
            data, hdr = nrrd.read(path)
            sp = hdr.get("spacings", hdr.get("space directions", None))
            if sp is None:
                spacing = (1.0, 1.0, 1.0)
            else:
                sp = np.array(sp, dtype=float)
                if sp.ndim == 2:
                    spacing = tuple(float(np.linalg.norm(sp[i])) for i in range(3))
                else:
                    spacing = tuple(float(v) for v in sp[:3])
        except ImportError:
            st.error("pynrrd not installed — rebuild the Docker image to add it.")
            return None, None
    else:
        import nibabel as nib
        nii     = nib.load(path)
        data    = nii.get_fdata(dtype=np.float32)
        spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32), spacing


def _lasc_infer(model, vol: np.ndarray, device: str, crop_size: int = 128):
    """Slice-by-slice TTA inference.  Returns full-resolution prediction map."""
    import torch, torch.nn.functional as F

    H, W, nz = vol.shape
    cx, cy   = H // 2, W // 2
    x0 = max(0, cx - crop_size // 2);  x1 = min(H, x0 + crop_size)
    y0 = max(0, cy - crop_size // 2);  y1 = min(W, y0 + crop_size)

    preds_full = np.zeros((H, W, nz), dtype=np.int32)

    for z in range(nz):
        slc = _percentile_norm(vol[x0:x1, y0:y1, z])
        t   = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
        _, _, h, w = t.shape
        nh = ((h + 31) // 32) * 32;  nw = ((w + 31) // 32) * 32
        tp = F.pad(t, (0, nw - w, 0, nh - h))
        with torch.no_grad():
            lg1, _ = model(tp)
            lg2, _ = model(torch.flip(tp, [3]))
            lg2    = torch.flip(lg2, [3])
            pred   = ((torch.softmax(lg1, 1) + torch.softmax(lg2, 1)) / 2
                      ).argmax(1).squeeze().cpu().numpy()
        preds_full[x0:x1, y0:y1, z] = pred[:h, :w]

    return preds_full


def _la_volumes(preds: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    vox_ml  = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    cav_ml  = float(np.sum(preds == 1) * vox_ml)
    wall_ml = float(np.sum(preds == 2) * vox_ml)
    return cav_ml, wall_ml


def _la_size_label(cav_ml: float):
    if cav_ml < 100:
        return "Normal",              "#2ECC71", "🟢", \
               "LA size is within normal range. Low AFib risk."
    elif cav_ml < 130:
        return "Mildly Enlarged",     "#F39C12", "🟡", \
               "Mild enlargement detected. Monitor for paroxysmal AFib."
    elif cav_ml < 160:
        return "Moderately Enlarged", "#E67E22", "🟠", \
               "Moderate enlargement. Elevated AFib risk. Consider further workup."
    else:
        return "Severely Enlarged",   "#E74C3C", "🔴", \
               "Severe enlargement. High AFib recurrence risk. Urgent cardiology review."


def _lasc_figure(vol: np.ndarray, preds: np.ndarray):
    """Matplotlib figure — 6 evenly-spaced LA slices with colour overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    nz = vol.shape[2]
    la_slices = [z for z in range(nz) if np.any(preds[:, :, z] > 0)]
    if not la_slices:
        la_slices = list(range(0, nz, max(1, nz // 6)))
    step   = max(1, len(la_slices) // 6)
    idxs   = la_slices[::step][:6]
    n_show = len(idxs)

    fig, axes = plt.subplots(
        1, n_show,
        figsize=(2.8 * n_show, 3.4),
        facecolor="#0d1117",
    )
    if n_show == 1:
        axes = [axes]

    for ax, z in zip(axes, idxs):
        raw = vol[:, :, z].astype(np.float32)
        fg  = raw[raw > 0]
        if len(fg):
            p1, p99 = np.percentile(fg, [1, 99])
            raw = np.clip((raw - p1) / max(p99 - p1, 1e-8), 0, 1)

        pred = preds[:, :, z]
        rgb  = np.stack([raw] * 3, axis=-1)
        for cls, col in LASC_COLORS_RGB.items():
            mask = pred == cls
            if mask.any():
                for c, v in enumerate(col):
                    rgb[:, :, c][mask] = 0.35 * raw[mask] + 0.65 * v

        ax.imshow(rgb)
        ax.set_title(f"slice {z}", color="#8899aa", fontsize=7.5, pad=4)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_visible(False)

    legend = [
        Patch(color=LASC_COLORS_RGB[1], label="LA Cavity"),
        Patch(color=LASC_COLORS_RGB[2], label="LA Wall"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=2,
        facecolor="#0d1117", edgecolor="#2a3f5f",
        labelcolor="white", fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Left Atrial Segmentation  ·  LGE-MRI",
        color="#1ABC9C", fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def _build_lasc_3d_html(preds: np.ndarray, spacing: tuple,
                        cav_ml: float, size_lbl: str, sz_col: str) -> str:
    """Build a self-contained HTML/Plotly 3D viewer for the LA segmentation.
    Includes the patient mesh (cavity + wall) AND a semi-transparent reference
    ghost mesh representing a ~100 ml 'Normal' LA for direct size comparison."""
    import json

    traces   = []
    meta     = {}

    # ── 1. Patient meshes (cavity + wall) ────────────────────────────────────
    layer_cfg = {
        1: {"color": "#6495ED", "name": "LA Cavity",  "opacity": 0.55, "sigma": 2.0},
        2: {"color": "#DC4F4F", "name": "LA Wall",    "opacity": 0.80, "sigma": 1.5},
    }
    global_center = np.zeros(3)
    n_centers = 0
    for cls, cfg in layer_cfg.items():
        verts, faces, _ = _extract_smooth_mesh(preds, cls,
                                               sigma=cfg["sigma"],
                                               spacing=spacing)
        if verts is None:
            continue
        verts, faces = _decimate(verts, faces, max_faces=18000)
        c = verts.mean(axis=0)
        global_center += c; n_centers += 1
        dists = np.linalg.norm(verts - c, axis=1)
        dists_n = (dists - dists.min()) / max(dists.max() - dists.min(), 1e-8)
        idx = len(traces)
        traces.append({
            "type": "mesh3d",
            "x": verts[:, 0].tolist(), "y": verts[:, 1].tolist(),
            "z": verts[:, 2].tolist(),
            "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(),
            "k": faces[:, 2].tolist(),
            "name": cfg["name"], "color": cfg["color"],
            "opacity": cfg["opacity"], "showlegend": True,
            "flatshading": False,
            "lighting": {"ambient": 0.4, "diffuse": 0.7,
                         "specular": 0.5, "roughness": 0.25, "fresnel": 0.3},
            "lightposition": {"x": 200, "y": 300, "z": 500},
            "hovertemplate": (f"<b>{cfg['name']}</b><br>"
                              "x:%{x:.0f} y:%{y:.0f} z:%{z:.0f}<extra></extra>"),
        })
        meta[str(idx)] = {"intensity_values": dists_n.tolist()}

    if n_centers:
        global_center /= n_centers

    # ── 2. Reference ghost — sphere scaled to 100 ml "Normal" LA ─────────────
    # Volume of sphere V = 4/3 π r³  →  r = (3V/4π)^(1/3)
    # 100 ml = 100 000 mm³
    ref_vol_mm3 = 100_000.0
    r_mm = (3 * ref_vol_mm3 / (4 * np.pi)) ** (1 / 3)  # ≈ 28.8 mm
    # Build icosphere-like mesh via UV parametrisation
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    uu, vv = np.meshgrid(u, v)
    sx = global_center[0] + r_mm * np.sin(vv) * np.cos(uu)
    sy = global_center[1] + r_mm * np.sin(vv) * np.sin(uu)
    sz = global_center[2] + r_mm * np.cos(vv)
    ref_faces_i, ref_faces_j, ref_faces_k = [], [], []
    nv, nu = sx.shape
    for ri in range(nv - 1):
        for ci in range(nu - 1):
            a = ri * nu + ci; b = a + 1
            c_ = (ri + 1) * nu + ci; d = c_ + 1
            ref_faces_i += [a, a]; ref_faces_j += [b, c_]; ref_faces_k += [c_, d]
    traces.append({
        "type": "mesh3d",
        "x": sx.ravel().tolist(), "y": sy.ravel().tolist(), "z": sz.ravel().tolist(),
        "i": ref_faces_i, "j": ref_faces_j, "k": ref_faces_k,
        "name": "Normal LA (100 ml ref.)", "color": "#aaaaaa",
        "opacity": 0.12, "showlegend": True, "flatshading": True,
        "hovertemplate": "<b>Normal LA reference</b><br>~100 ml<extra></extra>",
    })
    ref_idx = len(traces) - 1
    meta[str(ref_idx)] = {"intensity_values": [0.5] * len(ref_faces_i)}

    # enlargement ratio badge for overlay
    ratio = cav_ml / 100.0
    ratio_str = f"{ratio:.2f}×"

    traces_json = json.dumps(traces)
    meta_json   = json.dumps(meta)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#eee;font-family:'Segoe UI',sans-serif;overflow:hidden}}
#plot{{width:100vw;height:100vh}}
#panel{{position:fixed;top:14px;left:14px;z-index:100;
  background:rgba(15,20,35,.93);border-radius:14px;padding:16px 18px;
  min-width:240px;border:1px solid rgba(255,255,255,.1);
  backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.6)}}
#panel h2{{font-size:15px;margin-bottom:4px;color:#1ABC9C;font-weight:700}}
#panel .sub{{font-size:11px;color:#6495ED;margin-bottom:12px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:12px;font-weight:700;margin-bottom:12px;
  background:{sz_col}22;border:1px solid {sz_col}88;color:{sz_col}}}
.sec{{font-size:10px;color:#556;text-transform:uppercase;
  letter-spacing:.08em;margin:10px 0 5px}}
.row{{display:flex;align-items:center;gap:8px;margin:5px 0}}
.row label{{font-size:12px;min-width:72px;color:#ccc}}
.row input[type=range]{{flex:1;accent-color:#6495ED;height:4px}}
.row .v{{font-size:11px;color:#8899aa;min-width:32px;text-align:right}}
.btn{{display:inline-block;padding:5px 11px;margin:2px 2px 2px 0;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);
  border-radius:6px;color:#ccc;font-size:11px;cursor:pointer;transition:all .18s}}
.btn:hover{{background:rgba(255,255,255,.14);color:#fff}}
.btn.on{{background:{sz_col}33;border-color:{sz_col}88;color:{sz_col}}}
.leg{{display:flex;align-items:center;gap:7px;margin:4px 0;font-size:12px;cursor:pointer}}
.dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0}}
.tog{{font-size:10px;color:#556;margin-left:auto}}
</style></head><body>
<div id="plot"></div>
<div id="panel">
  <h2>&#x2764; Left Atrial 3D</h2>
  <div class="sub">Patient vs. Normal LA (grey ghost)</div>
  <div class="badge">{size_lbl} &nbsp;·&nbsp; {ratio_str} normal</div>

  <div class="sec">View Mode</div>
  <div>
    <span class="btn on"  id="b-solid"   onclick="setMode('solid')">Solid</span>
    <span class="btn"     id="b-xray"    onclick="setMode('xray')">X-Ray</span>
    <span class="btn"     id="b-heatmap" onclick="setMode('heatmap')">Heatmap</span>
  </div>

  <div class="sec">Controls</div>
  <div class="row"><label>Opacity</label>
    <input type="range" id="sl-op" min="5" max="100" value="70"
           oninput="updateOp(this.value)">
    <span class="v" id="v-op">70%</span></div>
  <div class="row"><label>Ref ghost</label>
    <input type="range" id="sl-ref" min="0" max="60" value="12"
           oninput="updateRef(this.value)">
    <span class="v" id="v-ref">12%</span></div>

  <div class="sec">Layers</div>
  <div id="legend"></div>
</div>
<script>
const traces={traces_json};
const meta={meta_json};
const refIdx={ref_idx};
let visMap={{}};
const layout={{
  scene:{{
    xaxis:{{visible:false}},yaxis:{{visible:false}},zaxis:{{visible:false}},
    aspectmode:'data',bgcolor:'rgb(13,17,23)',
    camera:{{eye:{{x:1.6,y:0.9,z:0.7}},up:{{x:0,y:0,z:1}}}}
  }},
  paper_bgcolor:'rgb(13,17,23)',margin:{{l:0,r:0,t:0,b:0}},showlegend:false
}};
Plotly.newPlot('plot',traces,layout,{{responsive:true,displayModeBar:false}});
// Build legend
const lg=document.getElementById('legend');
traces.forEach((t,i)=>{{
  visMap[i]=true;
  const d=document.createElement('div'); d.className='leg';
  d.innerHTML=`<div class="dot" style="background:${{t.color}};opacity:${{t.opacity}}"></div>
    <span>${{t.name}}</span><span class="tog" id="tog-${{i}}">ON</span>`;
  d.onclick=()=>toggle(i); lg.appendChild(d);
}});
function toggle(i){{
  visMap[i]=!visMap[i];
  Plotly.restyle('plot',{{visible:visMap[i]}},[i]);
  const el=document.getElementById('tog-'+i);
  el.textContent=visMap[i]?'ON':'OFF';
  el.style.color=visMap[i]?'#88ff88':'#ff5555';
}}
function updateOp(v){{
  document.getElementById('v-op').textContent=v+'%';
  const op=v/100;
  traces.forEach((t,i)=>{{
    if(i===refIdx) return;
    Plotly.restyle('plot',{{opacity:[op*t.opacity*1.4]}},[i]);
  }});
}}
function updateRef(v){{
  document.getElementById('v-ref').textContent=v+'%';
  Plotly.restyle('plot',{{opacity:[v/100]}},[refIdx]);
}}
let mode='solid';
function setMode(m){{
  mode=m;
  ['solid','xray','heatmap'].forEach(x=>document.getElementById('b-'+x).className='btn');
  document.getElementById('b-'+m).className='btn on';
  if(m==='solid'){{
    traces.forEach((t,i)=>{{
      if(i===refIdx) return;
      Plotly.restyle('plot',{{color:t.color,intensity:null,
        opacity:t.opacity,showscale:false}},[i]);
    }});
  }} else if(m==='xray'){{
    traces.forEach((t,i)=>{{
      if(i===refIdx) return;
      Plotly.restyle('plot',{{color:t.color,intensity:null,
        opacity:0.13,showscale:false}},[i]);
    }});
  }} else {{
    traces.forEach((t,i)=>{{
      if(i===refIdx) return;
      Plotly.restyle('plot',{{
        intensity:[meta[String(i)].intensity_values],
        colorscale:'Portland',color:null,opacity:0.9,showscale:i===0
      }},[i]);
    }});
  }}
}}
</script></body></html>"""


def _render_lasc_tab():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <h4 style='color:#1ABC9C;margin:0'>
        LASC 2018 — Left Atrial Segmentation &amp; Volume Analysis
    </h4>""", unsafe_allow_html=True)

    badges = (
        _model_badge("Model",    "ResBlock U-Net + ASPP", "#1ABC9C") +
        _model_badge("Input",    "LGE-MRI  (.nrrd / .nii)", "#9B59B6") +
        _model_badge("Classes",  "LA Cavity · LA Wall",   "#3498DB") +
        _model_badge("Task",     "Segmentation + Volume", "#E67E22")
    )
    st.markdown(f"<p style='margin:.45rem 0 1.1rem'>{badges}</p>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#BDC3C7;font-size:.9rem;margin-bottom:1.2rem'>
    Upload a Late Gadolinium Enhancement MRI (LGE-MRI) volume to automatically segment
    the <b style='color:#6495ED'>Left Atrial cavity</b> and
    <b style='color:#DC4F4F'>LA wall</b>, compute EDV-equivalent volumes and assess
    enlargement severity — a key predictor of <b>atrial fibrillation (AFib)</b> risk.
    </p>""", unsafe_allow_html=True)

    # ── Model availability check ─────────────────────────────────────────────
    if not os.path.isfile(LASC_MODEL):
        st.warning(
            f"⚠️ Model weights not found at `{LASC_MODEL}`. "
            "Copy `lasc_best.pth` into the `models_seg_dig/` folder and rebuild the image.",
            icon="⚠️",
        )
        return

    # ── File uploader ────────────────────────────────────────────────────────
    lge_file = st.file_uploader(
        "Upload LGE-MRI volume",
        type=["nrrd", "nii", "gz"],
        help="Accepts .nrrd (LASC native) or .nii / .nii.gz (NIfTI)",
    )

    run = st.button("▶ Run LASC Segmentation", type="primary",
                    disabled=lge_file is None)

    if not run or lge_file is None:
        # ── Info panel when idle ─────────────────────────────────────────────
        components.html("""
        <div style='background:#1a2535;border:1px solid #2a3f5f;border-radius:12px;
                    padding:1.3rem 1.5rem;font-family:sans-serif;margin-top:.5rem'>
          <div style='color:#1ABC9C;font-weight:700;font-size:.95rem;
                      margin-bottom:.9rem'>📐 LA Volume Reference Ranges</div>
          <div style='display:flex;gap:.7rem'>
            <div style='flex:1;background:#1e3a1e;border:1px solid #2ECC7155;
                        border-radius:8px;padding:.8rem;text-align:center'>
              <div style='color:#2ECC71;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.07em'>Normal</div>
              <div style='color:#ECF0F1;font-size:1.5rem;font-weight:800;
                          line-height:1.1;margin:.3rem 0'>&lt; 100 ml</div>
              <div style='color:#8899aa;font-size:.7rem'>Low AFib risk</div>
            </div>
            <div style='flex:1;background:#3a2f10;border:1px solid #F39C1255;
                        border-radius:8px;padding:.8rem;text-align:center'>
              <div style='color:#F39C12;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.07em'>Mild</div>
              <div style='color:#ECF0F1;font-size:1.5rem;font-weight:800;
                          line-height:1.1;margin:.3rem 0'>100–130 ml</div>
              <div style='color:#8899aa;font-size:.7rem'>Monitor for AFib</div>
            </div>
            <div style='flex:1;background:#3a2010;border:1px solid #E67E2255;
                        border-radius:8px;padding:.8rem;text-align:center'>
              <div style='color:#E67E22;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.07em'>Moderate</div>
              <div style='color:#ECF0F1;font-size:1.5rem;font-weight:800;
                          line-height:1.1;margin:.3rem 0'>130–160 ml</div>
              <div style='color:#8899aa;font-size:.7rem'>Elevated risk</div>
            </div>
            <div style='flex:1;background:#3a1010;border:1px solid #E74C3C55;
                        border-radius:8px;padding:.8rem;text-align:center'>
              <div style='color:#E74C3C;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.07em'>Severe</div>
              <div style='color:#ECF0F1;font-size:1.5rem;font-weight:800;
                          line-height:1.1;margin:.3rem 0'>&gt; 160 ml</div>
              <div style='color:#8899aa;font-size:.7rem'>High AFib recurrence</div>
            </div>
          </div>
        </div>""", height=195)
        return

    # ── Run inference ────────────────────────────────────────────────────────
    tmp_path = None
    try:
        suffix   = ".nrrd" if lge_file.name.endswith(".nrrd") else \
                   (".nii.gz" if lge_file.name.endswith(".gz") else ".nii")
        tmp      = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        lge_file.seek(0)
        tmp.write(lge_file.read())
        tmp.close()
        tmp_path = tmp.name

        with st.spinner("Loading volume…"):
            vol, spacing = _lasc_load_volume(tmp_path)
        if vol is None:
            return

        with st.spinner("Segmenting left atrium slice-by-slice (TTA)…"):
            model, device = _load_lasc()
            preds         = _lasc_infer(model, vol, device)

        cav_ml, wall_ml          = _la_volumes(preds, spacing)
        size_lbl, sz_col, sz_ico, sz_note = _la_size_label(cav_ml)
        la_slices = [z for z in range(vol.shape[2]) if np.any(preds[:, :, z] > 0)]

        # ── Volume metric cards ──────────────────────────────────────────────
        components.html(f"""
        <div style='background:#1a2535;border:1px solid #2a3f5f;border-radius:12px;
                    padding:1.3rem 1.5rem;font-family:sans-serif;margin-bottom:.5rem'>
          <div style='color:#1ABC9C;font-weight:700;font-size:.95rem;
                      margin-bottom:.85rem'>📊 Cardiac Volumes</div>
          <div style='display:flex;gap:.75rem;margin-bottom:.9rem'>

            <!-- Cavity -->
            <div style='flex:1;background:#273241;border-radius:10px;
                        padding:.85rem;text-align:center'>
              <div style='color:#6495ED;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.08em;
                          margin-bottom:.25rem'>LA Cavity</div>
              <div style='color:#ECF0F1;font-size:2rem;font-weight:800;
                          line-height:1'>{cav_ml:.1f}</div>
              <div style='color:#5D6D7E;font-size:.7rem;margin-top:.2rem'>
                mL &nbsp;·&nbsp; <span style='color:#4a6070'>norm &lt; 100</span>
              </div>
            </div>

            <!-- Wall -->
            <div style='flex:1;background:#273241;border-radius:10px;
                        padding:.85rem;text-align:center'>
              <div style='color:#DC4F4F;font-size:.72rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:.08em;
                          margin-bottom:.25rem'>LA Wall</div>
              <div style='color:#ECF0F1;font-size:2rem;font-weight:800;
                          line-height:1'>{wall_ml:.1f}</div>
              <div style='color:#5D6D7E;font-size:.7rem;margin-top:.2rem'>mL</div>
            </div>

            <!-- Size classification -->
            <div style='flex:2;background:{sz_col}18;border:1.5px solid {sz_col}66;
                        border-radius:10px;padding:.85rem;text-align:center'>
              <div style='color:{sz_col};font-size:.78rem;font-weight:700;
                          margin-bottom:.2rem'>LA Enlargement {sz_ico}</div>
              <div style='color:{sz_col};font-size:1.6rem;font-weight:800;
                          line-height:1.1'>{size_lbl}</div>
              <div style='color:#BDC3C7;font-size:.72rem;margin-top:.3rem'>
                {len(la_slices)} slices with LA tissue
              </div>
            </div>

          </div>

          <!-- Progress bar: cavity size -->
          <div style='background:#273241;border-radius:6px;height:9px;
                      overflow:hidden;margin-bottom:.18rem'>
            <div style='width:{min(cav_ml/200*100, 100):.1f}%;
                        background:linear-gradient(90deg,#2ECC71 0%,#2ECC71 49%,
                          #F39C12 50%,#E67E22 65%,#E74C3C 80%,#E74C3C 100%);
                        height:100%;border-radius:6px'></div>
          </div>
          <div style='display:flex;justify-content:space-between;
                      color:#4a6070;font-size:.65rem;margin-bottom:1rem'>
            <span>0 mL</span>
            <span style='color:#2ECC71'>100 mL</span>
            <span style='color:#F39C12'>130 mL</span>
            <span style='color:#E67E22'>160 mL</span>
            <span style='color:#E74C3C'>200+ mL</span>
          </div>

          <!-- Clinical note -->
          <div style='background:{sz_col}15;border-left:3px solid {sz_col};
                      border-radius:0 6px 6px 0;padding:.6rem .9rem'>
            <span style='color:{sz_col};font-weight:700'>⚕ Clinical Note: </span>
            <span style='color:#BDC3C7'>{sz_note}</span>
          </div>
        </div>""", height=365)

        # ── 3D interactive viewer ────────────────────────────────────────────
        st.markdown(
            "<p style='color:#8899aa;font-size:.82rem;margin:.8rem 0 .3rem'>"
            "🫀 Interactive 3D — patient LA vs. normal reference (grey ghost)</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("Building 3D mesh…"):
            html_3d = _build_lasc_3d_html(preds, spacing, cav_ml, size_lbl, sz_col)
        components.html(html_3d, height=520, scrolling=False)

        # ── Slice visualisation ──────────────────────────────────────────────
        st.markdown(
            "<p style='color:#8899aa;font-size:.82rem;margin:.6rem 0 .3rem'>"
            "🔬 Segmentation overlay — 6 representative LGE-MRI slices</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("Rendering slice preview…"):
            fig = _lasc_figure(vol, preds)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150,
                        facecolor="#0d1117", bbox_inches="tight")
            buf.seek(0)
            import matplotlib.pyplot as plt
            plt.close(fig)

        st.image(buf, use_container_width=True)

        # ── Download segmentation ────────────────────────────────────────────
        seg_buf = io.BytesIO()
        np.save(seg_buf, preds)
        seg_buf.seek(0)
        st.download_button(
            "⬇️ Download segmentation mask (.npy)",
            data=seg_buf,
            file_name=f"lasc_seg_{lge_file.name.split('.')[0]}.npy",
            mime="application/octet-stream",
        )

        # ── Disclaimer ───────────────────────────────────────────────────────
        st.markdown("""
        <div style='background:#2c1a0a;border-radius:8px;padding:.7rem 1rem;
                    margin-top:1rem;border-left:4px solid #E67E22'>
          <span style='color:#E67E22;font-weight:700'>⚕ Clinical Disclaimer</span><br>
          <span style='color:#BDC3C7;font-size:.875rem'>Research tool only. Not validated for
          clinical use. All decisions must be made by qualified medical professionals.</span>
        </div>""", unsafe_allow_html=True)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            _safe_unlink(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def render():
    st.markdown(
        "<h2 style='color:#1ABC9C'>Heart Reconstruction &amp; Diagnosis</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#BDC3C7'>End-to-end cardiac AI pipeline — whole-heart 3D "
        "reconstruction from NIfTI volumes and deep-learning disease diagnosis "
        "from MRI sequences.</p>",
        unsafe_allow_html=True,
    )

    tab_recon, tab_diag = st.tabs([
        "🫀 Segmentation & 3D Heart Reconstruction",
        "🩺 Cardiac Diagnosis",
    ])

    with tab_recon:
        _render_mmwhs_tab()

    with tab_diag:
        st.markdown("""
        <p style='color:#BDC3C7;margin-bottom:1rem'>
        Three complementary diagnostic tools:<br>
        <b>ACDC</b> — pathology classification (Normal / MINF / DCM / HCM / ARV / HHD)
        from single short-axis MRI frames, with optional EF estimation.<br>
        <b>MyoPS</b> — scar &amp; edema detection and quantification from three
        co-registered sequences (C0 + DE + T2).<br>
        <b>LASC</b> — left atrial cavity &amp; wall segmentation from LGE-MRI with
        volume-based AFib risk assessment.
        </p>""", unsafe_allow_html=True)

        sub_acdc, sub_myops, sub_lasc = st.tabs([
            "📋 ACDC — Disease Classification",
            "🔬 MyoPS — Scar & Edema",
            "🫀 LASC — Left Atrial Analysis",
        ])
        with sub_acdc:
            _render_acdc_tab()
        with sub_myops:
            _render_myops_tab()
        with sub_lasc:
            _render_lasc_tab()
