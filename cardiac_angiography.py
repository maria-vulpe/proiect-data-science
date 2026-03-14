"""
Cardiac Angiography Page
========================
Integrates two coronary artery analysis pipelines:

  ARCADE  – Mask R-CNN instance segmentation for coronary vessel
            classification (25 vessel segments) and stenosis detection.
            Reproduces: Popov et al., 2024 (Zenodo 10390295).

  CADICA  – Three-stage CAD detection pipeline:
            ResNet-50 lesion detector → ResNet-18 stenosis estimator
            → Gradient Boosting classifier.
"""

import os
import io
import json
import pickle
import tempfile

import numpy as np
import streamlit as st
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file, suffix: str) -> str:
    """Write an UploadedFile to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ARCADE helpers (adapted from arcade_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _arcade_build_model(num_classes: int):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes)
    return model


def _arcade_enhance_xca(bgr):
    """White top-hat on negative + CLAHE preprocessing (paper baseline)."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((50, 50), np.uint8)
    tophat = cv2.morphologyEx(255 - gray, cv2.MORPH_TOPHAT, kernel)
    enh = np.clip(gray.astype(np.int16) - tophat.astype(np.int16), 0, 255).astype(np.uint8)
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enh)
    return cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)


def _arcade_predict(model, pil_image: Image.Image, task: str,
                    score_thr: float, enhance: bool, device: str):
    """Run ARCADE Mask R-CNN inference. Returns (result_pil, detections_list)."""
    import torch, cv2
    from torchvision import transforms as T

    SYN_NAMES = {
        1: "RCA prox", 2: "RCA mid", 3: "RCA dist", 4: "Post desc",
        5: "Left main", 6: "LAD prox", 7: "LAD mid", 8: "LAD dist",
        9: "1st diag", 10: "2nd diag", 11: "Prox circ", 12: "Intermed",
        13: "Dist circ", 14: "L postlat", 15: "PD(LCX)", 16: "PL(RCA)",
        17: "9a", 18: "12a", 19: "12b", 20: "14a", 21: "14b",
        22: "16a", 23: "16b", 24: "16c", 25: "Ramus",
    }
    names = SYN_NAMES if task == "vessel" else {1: "stenosis"}

    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if enhance:
        bgr = _arcade_enhance_xca(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ten = T.ToTensor()(rgb).unsqueeze(0).to(device)

    model.eval().to(device)
    with torch.no_grad():
        pred = model(ten)[0]

    all_scores = pred["scores"].cpu().numpy()
    keep = pred["scores"].cpu() > score_thr
    boxes  = pred["boxes"][keep].cpu().numpy()
    labels = pred["labels"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()
    masks  = pred["masks"][keep].cpu().numpy()

    # Draw
    np.random.seed(42)
    pal = [(np.random.randint(80, 255), np.random.randint(80, 255),
            np.random.randint(80, 255)) for _ in range(30)]
    vis = bgr.copy()
    for i in range(len(boxes)):
        c = pal[int(labels[i]) % len(pal)]
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        label_txt = f"{names.get(int(labels[i]), labels[i])} {scores[i]:.2f}"
        cv2.putText(vis, label_txt, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        if i < len(masks):
            m = (masks[i][0] > 0.5).astype(np.uint8)
            ov = np.zeros_like(vis)
            ov[:, :] = c
            vis = np.where(m[..., None], cv2.addWeighted(vis, 0.5, ov, 0.5, 0), vis)

    result_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    detections = []
    for i in range(len(boxes)):
        detections.append({
            "label": names.get(int(labels[i]), f"cls{labels[i]}"),
            "score": float(scores[i]),
            "box": boxes[i].astype(int).tolist(),
        })

    return result_pil, detections, all_scores


# ─────────────────────────────────────────────────────────────────────────────
# CADICA helpers (adapted from cadica_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def _cadica_load_lesion_model(path: str, device: str):
    import torch
    import torch.nn as nn
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval().to(device)


def _cadica_load_stenosis_model(path: str, device: str):
    import torch
    import torch.nn as nn
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 3)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval().to(device)


def _cadica_detect_lesion(model, img_tensor, device):
    import torch
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        return bool(probs[1] > 0.5), float(probs[1])


def _cadica_estimate_stenosis(stenosis_model, img_pil: Image.Image,
                              crop_transform, device):
    import torch
    if stenosis_model is None:
        return 1, 0.5
    img_t = crop_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = stenosis_model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
        return int(probs.argmax()), float(probs.max())


def _cadica_predict_images(images, lesion_model, stenosis_model,
                           pipeline, device, frame_transform, crop_transform):
    """Core CADICA prediction logic. Returns result dict."""
    from collections import defaultdict

    frame_results = []
    for img in images:
        img_t = frame_transform(img)
        is_lesion, lesion_conf = _cadica_detect_lesion(lesion_model, img_t, device)
        severities = []
        for _ in range(3):
            sev, _ = _cadica_estimate_stenosis(stenosis_model, img, crop_transform, device)
            severities.append(sev)
        severity = int(np.median(severities))
        frame_results.append({
            "is_lesion": is_lesion,
            "lesion_confidence": lesion_conf,
            "severity": severity if is_lesion else 0,
        })

    total = len(frame_results)
    lesion_count = sum(1 for r in frame_results if r["is_lesion"])

    STENOSIS_CATS = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

    if total >= 10 and pipeline is not None:
        # Build feature vector
        lesion_ratio = lesion_count / max(total, 1)
        severity_counts = defaultdict(int)
        confs = []
        for r in frame_results:
            if r["is_lesion"]:
                severity_counts[r["severity"]] += 1
                confs.append(r["lesion_confidence"])

        mild = severity_counts.get(0, 0)
        moderate = severity_counts.get(1, 0)
        severe = severity_counts.get(2, 0)
        sten_approx = {
            "p0_20": mild // 2, "p20_50": mild - mild // 2,
            "p50_70": moderate,
            "p70_90": severe // 3, "p90_98": severe // 3,
            "p99": severe - 2 * (severe // 3), "p100": 0,
        }
        sten_vec = [sten_approx.get(c, 0) for c in STENOSIS_CATS]
        sten_frac = [s / max(total, 1) for s in sten_vec]
        avg_sev = np.mean([r["severity"] for r in frame_results if r["is_lesion"]]) if lesion_count else 0
        max_sev = max((r["severity"] for r in frame_results if r["is_lesion"]), default=0)
        avg_conf = np.mean(confs) if confs else 0
        std_conf = np.std(confs) if len(confs) > 1 else 0
        area_est = (avg_sev + 1) * 300 if lesion_count else 0
        area_max = (max_sev + 1) * 600 if lesion_count else 0

        features = np.array([
            total, lesion_count, lesion_ratio,
            lesion_count,
            lesion_count / max(total, 1),
            1.0 if lesion_count > 0 else 0,
            max(1, lesion_count // max(total // 5, 1)) if total >= 5 else (1 if lesion_count > 0 else 0),
            max(0, (total // 5) - (lesion_count // max(total // 5, 1))) if total >= 5 else (0 if lesion_count > 0 else 1),
            lesion_ratio,
        ] + sten_vec + sten_frac
          + [area_est, area_est * max(std_conf, 0.3), area_max, area_est * 0.8]
          + [max(area_est, 1) ** 0.5 * 0.7, max(area_max, 1) ** 0.5]
          + [max(area_est, 1) ** 0.5 * 0.5, max(area_max, 1) ** 0.5 * 0.8]
          + [avg_sev * 2, max_sev * 3],
            dtype=np.float32)

        X = features.reshape(1, -1)
        X_cad = pipeline["scaler_cad"].transform(X)
        cad_pred = pipeline["clf_cad"].predict(X_cad)[0]
        cad_proba = pipeline["clf_cad"].predict_proba(X_cad)[0]
        X_v = pipeline["scaler_vessel"].transform(X)
        vessel_pred = pipeline["clf_vessel"].predict(X_v)[0]
        vessel_proba = pipeline["clf_vessel"].predict_proba(X_v)[0]
        method = "Gradient Boosting (multi-frame)"
        result = {
            "has_cad": bool(cad_pred),
            "cad_confidence": float(max(cad_proba)),
            "cad_probabilities": {"No CAD": float(cad_proba[0]), "Has CAD": float(cad_proba[1])},
            "num_vessels": int(vessel_pred),
            "vessel_confidence": float(max(vessel_proba)),
            "vessel_probabilities": {f"{i}v": float(vessel_proba[i]) for i in range(len(vessel_proba))},
        }
    else:
        # CNN-direct fallback
        lesion_ratio = lesion_count / max(total, 1)
        severities_lesion = [r["severity"] for r in frame_results if r["is_lesion"]]
        avg_sev = np.mean(severities_lesion) if severities_lesion else 0
        max_sev = max(severities_lesion) if severities_lesion else 0
        if lesion_ratio > 0.3 and avg_sev >= 1.5:
            cad_prob = min(0.95, 0.5 + lesion_ratio * 0.3 + avg_sev * 0.1)
        elif lesion_ratio > 0.5:
            cad_prob = min(0.85, 0.4 + lesion_ratio * 0.3)
        elif lesion_count > 0 and avg_sev >= 1:
            cad_prob = 0.4 + avg_sev * 0.15
        else:
            cad_prob = max(0.05, lesion_ratio * 0.3)
        has_cad = cad_prob > 0.5
        if not has_cad:
            vp = [0.8, 0.15, 0.04, 0.01]
        elif max_sev <= 0 and avg_sev < 0.5:
            vp = [0.3, 0.5, 0.15, 0.05]
        elif max_sev == 1 or (0.5 <= avg_sev < 1.5):
            vp = [0.05, 0.55, 0.3, 0.1]
        elif max_sev >= 2 and avg_sev >= 1.5:
            vp = [0.02, 0.2, 0.4, 0.38]
        else:
            vp = [0.1, 0.45, 0.3, 0.15]
        method = "CNN-Direct (single/few frames)"
        result = {
            "has_cad": has_cad,
            "cad_confidence": float(max(cad_prob, 1 - cad_prob)),
            "cad_probabilities": {"No CAD": float(1 - cad_prob), "Has CAD": float(cad_prob)},
            "num_vessels": int(np.argmax(vp)),
            "vessel_confidence": float(max(vp)),
            "vessel_probabilities": {f"{i}v": float(vp[i]) for i in range(4)},
        }

    avg_severity_overall = np.mean([r["severity"] for r in frame_results if r["is_lesion"]]) if lesion_count else 0
    severity_map = {0: "No significant CAD", 1: "Single-vessel disease",
                    2: "Double-vessel disease", 3: "Triple-vessel disease"}
    result.update({
        "vessel_description": severity_map.get(result["num_vessels"], "Unknown"),
        "frames_analyzed": total,
        "lesion_frames": lesion_count,
        "lesion_ratio": lesion_count / max(total, 1),
        "average_severity": float(avg_severity_overall),
        "method": method,
        "frame_details": frame_results,
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ARCADE UI tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_arcade_tab():
    st.markdown("""
    <div style='background:#273241;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem;'>
    <h4 style='color:#1ABC9C;margin:0'>ARCADE — Coronary Vessel & Stenosis Segmentation</h4>
    <p style='color:#BDC3C7;margin:0.4rem 0 0'>
    Mask R-CNN (ResNet-50 FPN v2) trained on the ARCADE dataset.<br>
    Classifies 25 coronary artery segments <em>or</em> detects stenosis lesions
    with instance-level segmentation masks.
    </p></div>
    """, unsafe_allow_html=True)

    col_cfg, col_img = st.columns([1, 1], gap="large")

    with col_cfg:
        st.markdown("#### Model & Settings")

        task = st.radio("Task", ["vessel", "stenosis"],
                        format_func=lambda t: "Vessel classification (25 segments)" if t == "vessel" else "Stenosis detection",
                        horizontal=True)

        model_file = st.file_uploader("Trained checkpoint (.pth)",
                                      type=["pth", "pt"],
                                      key="arcade_model_file",
                                      help="Upload vessel_best.pth or stenosis_best.pth")

        score_thr = st.slider("Score threshold", 0.05, 0.95, 0.30, 0.05,
                              key="arcade_score_thr",
                              help="Detections below this confidence are hidden")

        enhance = st.checkbox("Apply XCA preprocessing",
                              key="arcade_enhance",
                              help="White top-hat + CLAHE enhancement from the paper")

        image_file = st.file_uploader("Angiography image",
                                      type=["png", "jpg", "jpeg", "bmp"],
                                      key="arcade_image")

    with col_img:
        st.markdown("#### Input Preview")
        if image_file:
            pil_img = Image.open(image_file).convert("RGB")
            st.image(pil_img, caption="Uploaded image", width="stretch")
        else:
            st.markdown("""
            <div style='background:#1E1E2F;border-radius:8px;padding:3rem;text-align:center;color:#7F8C8D;'>
            Upload an angiography image to preview it here.
            </div>""", unsafe_allow_html=True)

    st.divider()

    run_btn = st.button("▶ Run ARCADE Inference", type="primary",
                        disabled=(model_file is None or image_file is None),
                        key="arcade_run")

    if model_file is None or image_file is None:
        st.info("Upload both a checkpoint and an image to enable inference.")
        return

    if not run_btn:
        return

    try:
        import torch
    except ImportError:
        st.error("PyTorch is not installed. Run: `pip install torch torchvision`")
        return

    try:
        import cv2
    except ImportError:
        st.error("OpenCV is not installed. Run: `pip install opencv-python`")
        return

    with st.spinner("Loading model & running inference…"):
        device = _get_device()
        nc = 26 if task == "vessel" else 2

        # Save model to temp file
        model_path = _save_upload(model_file, ".pth")
        try:
            model = _arcade_build_model(nc)
            ckpt = torch.load(model_path, map_location=device)
            state = ckpt.get("state", ckpt)
            model.load_state_dict(state)
            epoch_info = ckpt.get("epoch", "?")
            val_loss_info = ckpt.get("val_loss", "?")
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            os.unlink(model_path)
            return
        finally:
            os.unlink(model_path)

        # Re-open image (file cursor may have moved)
        image_file.seek(0)
        pil_img = Image.open(image_file).convert("RGB")

        try:
            result_pil, detections, all_scores = _arcade_predict(
                model, pil_img, task, score_thr, enhance, device)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

    # Results
    st.success(f"Inference complete — device: {device.upper()} | "
               f"checkpoint epoch: {epoch_info} | val_loss: "
               f"{val_loss_info:.4f}" if isinstance(val_loss_info, float) else
               f"Inference complete — device: {device.upper()}")

    res_col, det_col = st.columns([3, 2], gap="large")

    with res_col:
        st.markdown("#### Segmentation Result")
        st.image(result_pil, caption="Predicted masks & bounding boxes",
                 width="stretch")
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("Download result image", buf.getvalue(),
                           file_name=f"arcade_{task}_prediction.png",
                           mime="image/png")

    with det_col:
        st.markdown("#### Detections")
        st.metric("Total raw detections", len(all_scores))
        st.metric(f"Above threshold ({score_thr})", len(detections))

        if len(all_scores):
            import pandas as pd
            score_df = pd.DataFrame({
                "Range": ["≥ 0.5", "0.3 – 0.5", "0.1 – 0.3", "< 0.1"],
                "Count": [
                    int((all_scores >= 0.5).sum()),
                    int(((all_scores >= 0.3) & (all_scores < 0.5)).sum()),
                    int(((all_scores >= 0.1) & (all_scores < 0.3)).sum()),
                    int((all_scores < 0.1).sum()),
                ]
            })
            st.dataframe(score_df, hide_index=True, width="stretch")

        if detections:
            st.markdown("**Detected regions:**")
            import pandas as pd
            det_df = pd.DataFrame([
                {"Segment": d["label"],
                 "Confidence": f"{d['score']:.2f}",
                 "Box [x1,y1,x2,y2]": str(d["box"])}
                for d in detections
            ])
            st.dataframe(det_df, hide_index=True, width="stretch")
        else:
            st.warning("No detections above threshold. Try lowering the score threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# CADICA UI tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_cadica_tab():
    st.markdown("""
    <div style='background:#273241;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem;'>
    <h4 style='color:#1ABC9C;margin:0'>CADICA — Coronary Artery Disease Detection</h4>
    <p style='color:#BDC3C7;margin:0.4rem 0 0'>
    Three-stage pipeline: ResNet-50 lesion detector → ResNet-18 stenosis
    severity estimator → Gradient Boosting CAD classifier.<br>
    Upload one frame for a quick CNN-based estimate, or multiple frames for
    the full Gradient Boosting pipeline (≥10 frames recommended).
    </p></div>
    """, unsafe_allow_html=True)

    # Model upload section
    st.markdown("#### Upload Trained Models")
    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        lesion_file = st.file_uploader("Lesion Detector (.pth)",
                                       type=["pth", "pt"],
                                       key="cadica_lesion",
                                       help="lesion_detector.pth")
    with mcol2:
        stenosis_file = st.file_uploader("Stenosis Estimator (.pth) — optional",
                                         type=["pth", "pt"],
                                         key="cadica_stenosis",
                                         help="stenosis_estimator.pth")
    with mcol3:
        clf_file = st.file_uploader("GB Classifiers (.pkl) — optional",
                                    type=["pkl"],
                                    key="cadica_clf",
                                    help="classifiers.pkl (needed for ≥10 frames)")

    st.divider()
    st.markdown("#### Upload Angiography Frames")
    st.caption("Upload one image for a quick single-frame estimate, or multiple frames for a full multi-frame analysis.")

    image_files = st.file_uploader("Angiography images (.png / .jpg)",
                                   type=["png", "jpg", "jpeg", "bmp"],
                                   accept_multiple_files=True,
                                   key="cadica_images")

    # Image preview
    if image_files:
        n_prev = min(len(image_files), 4)
        prev_cols = st.columns(n_prev)
        for i, col in enumerate(prev_cols):
            with col:
                img = Image.open(image_files[i]).convert("RGB")
                col.image(img, caption=image_files[i].name,
                          width="stretch")
        if len(image_files) > 4:
            st.caption(f"… and {len(image_files) - 4} more frame(s)")

    # Optional clinical info
    with st.expander("Optional: Patient Clinical Information"):
        ccol1, ccol2, ccol3 = st.columns(3)
        with ccol1:
            age = st.number_input("Age (years)", 18, 120, 65, key="cadica_age")
        with ccol2:
            sex = st.selectbox("Sex", ["M", "F"], key="cadica_sex")
        with ccol3:
            diabetes = st.selectbox("Diabetes mellitus", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No",
                                    key="cadica_diabetes")

    st.divider()

    run_btn = st.button("▶ Run CADICA Analysis", type="primary",
                        disabled=(lesion_file is None or not image_files),
                        key="cadica_run")

    if lesion_file is None:
        st.info("Upload at least the Lesion Detector model to enable analysis.")
        return

    if not image_files:
        st.info("Upload one or more angiography frames to enable analysis.")
        return

    if not run_btn:
        return

    try:
        import torch
        from torchvision import transforms as T
    except ImportError:
        st.error("PyTorch is not installed. Run: `pip install torch torchvision`")
        return

    with st.spinner("Loading models…"):
        device = _get_device()

        # Lesion detector
        lesion_path = _save_upload(lesion_file, ".pth")
        try:
            lesion_model = _cadica_load_lesion_model(lesion_path, device)
        except Exception as e:
            st.error(f"Failed to load lesion detector: {e}")
            os.unlink(lesion_path)
            return
        finally:
            os.unlink(lesion_path)

        # Stenosis estimator
        stenosis_model = None
        if stenosis_file:
            sten_path = _save_upload(stenosis_file, ".pth")
            try:
                stenosis_model = _cadica_load_stenosis_model(sten_path, device)
            except Exception as e:
                st.warning(f"Could not load stenosis estimator: {e}. Continuing without it.")
            finally:
                os.unlink(sten_path)

        # Gradient boosting classifiers
        pipeline = None
        if clf_file:
            try:
                pipeline = pickle.load(clf_file)
            except Exception as e:
                st.warning(f"Could not load classifiers: {e}. Will use CNN-direct mode.")

        frame_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        crop_transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    with st.spinner(f"Analyzing {len(image_files)} frame(s)…"):
        images = []
        for f in image_files:
            f.seek(0)
            images.append(Image.open(f).convert("RGB"))

        try:
            result = _cadica_predict_images(
                images, lesion_model, stenosis_model,
                pipeline, device, frame_transform, crop_transform)
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return

    # ── Results ──────────────────────────────────────────────────────────────
    st.success(f"Analysis complete | Method: **{result['method']}** | Device: {device.upper()}")

    # Summary cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    cad_color = "#E74C3C" if result["has_cad"] else "#2ECC71"
    cad_label = "⚠ CAD DETECTED" if result["has_cad"] else "✓ No CAD"
    sc1.markdown(f"""
    <div class='feature-card'>
    <h4 style='color:{cad_color}'>{cad_label}</h4>
    <p>Confidence: {result['cad_confidence']:.0%}</p>
    </div>""", unsafe_allow_html=True)

    sc2.markdown(f"""
    <div class='feature-card'>
    <h4>{result['num_vessels']} vessel(s)</h4>
    <p>{result['vessel_description']}</p>
    </div>""", unsafe_allow_html=True)

    sc3.markdown(f"""
    <div class='feature-card'>
    <h4>{result['lesion_frames']} / {result['frames_analyzed']}</h4>
    <p>Frames with lesions ({result['lesion_ratio']:.0%})</p>
    </div>""", unsafe_allow_html=True)

    SEVERITY_NAMES = ["Mild (<50%)", "Moderate (50–70%)", "Severe (>70%)"]
    avg_sev_idx = min(int(round(result["average_severity"])), 2)
    sc4.markdown(f"""
    <div class='feature-card'>
    <h4>{SEVERITY_NAMES[avg_sev_idx]}</h4>
    <p>Avg stenosis severity</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bars
    prob_col, vessel_col = st.columns(2)

    with prob_col:
        st.markdown("**CAD probability**")
        import altair as alt
        import pandas as pd

        cad_df = pd.DataFrame([
            {"Category": k, "Probability": v}
            for k, v in result["cad_probabilities"].items()
        ])
        cad_chart = (
            alt.Chart(cad_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Category:N", axis=alt.Axis(labelColor="#ECF0F1", titleColor="#1ABC9C")),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(labelColor="#ECF0F1", titleColor="#1ABC9C")),
                color=alt.Color("Category:N",
                                scale=alt.Scale(domain=["No CAD", "Has CAD"],
                                                range=["#2ECC71", "#E74C3C"]),
                                legend=None),
                tooltip=["Category", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=200, background="#273241")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(cad_chart, width="stretch")

    with vessel_col:
        st.markdown("**Vessels affected — probability**")
        vessel_df = pd.DataFrame([
            {"Vessels": k, "Probability": v}
            for k, v in result["vessel_probabilities"].items()
        ])
        vessel_chart = (
            alt.Chart(vessel_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Vessels:N", axis=alt.Axis(labelColor="#ECF0F1", titleColor="#1ABC9C")),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(labelColor="#ECF0F1", titleColor="#1ABC9C")),
                color=alt.Color("Probability:Q",
                                scale=alt.Scale(scheme="tealblues"),
                                legend=None),
                tooltip=["Vessels", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=200, background="#273241")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(vessel_chart, width="stretch")

    # Per-frame details
    with st.expander("Per-frame lesion details"):
        frame_df = pd.DataFrame([
            {
                "Frame": image_files[i].name if i < len(image_files) else f"Frame {i+1}",
                "Lesion detected": "Yes" if r["is_lesion"] else "No",
                "Confidence": f"{r['lesion_confidence']:.2%}",
                "Severity": SEVERITY_NAMES[min(r["severity"], 2)] if r["is_lesion"] else "—",
            }
            for i, r in enumerate(result["frame_details"])
        ])
        st.dataframe(frame_df, hide_index=True, width="stretch")

    # Disclaimer
    st.markdown("""
    <div style='background:#1E1E2F;border-radius:8px;padding:0.75rem 1rem;margin-top:1rem;
                border-left:4px solid #E67E22;'>
    <span style='color:#E67E22;font-weight:700'>⚕ Clinical Disclaimer</span><br>
    <span style='color:#BDC3C7;font-size:0.875rem'>
    This is a computer-aided detection tool only. All clinical decisions must be
    made by qualified medical professionals. Results are not a substitute for
    expert interpretation of coronary angiography.
    </span></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main render function (entry point called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        "<h2 style='color:#1ABC9C;'>Cardiac Angiography Analysis</h2>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#BDC3C7;'>Deep-learning pipelines for coronary artery "
        "analysis — vessel segmentation, stenosis detection, and CAD classification.</p>",
        unsafe_allow_html=True)

    tab_arcade, tab_cadica = st.tabs([
        "🫀 ARCADE — Vessel Segmentation",
        "🩺 CADICA — CAD Detection",
    ])

    with tab_arcade:
        _render_arcade_tab()

    with tab_cadica:
        _render_cadica_tab()
