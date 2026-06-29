import os
import io
import base64
from typing import List, Tuple

import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# === import your existing model class ===
from model_utils import estModel

# ---------------- Configuration ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_100epocs.pth")
YAML_PATH  = os.environ.get("YAML_PATH",  "data.yaml")
OUT_DIR    = os.environ.get("OUT_DIR",    "outputs_api")
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("THRESHOLD", "0.3"))  # same as your script
USE_CPU = os.environ.get("FORCE_CPU", "0") == "1"  # set to 1 to force CPU

# Optional: very simple Basic Auth (user preference in your notes)
BASIC_USER = os.environ.get("BASIC_USER")
BASIC_PASS = os.environ.get("BASIC_PASS")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Utils (match your existing pipeline) ----------------
def read_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        contents = yaml.safe_load(f)
    return contents["nc"], contents["names"]

def to_tensor_chw_uint(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR -> RGB -> CHW float32 [0,1] (same normalization you used)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.tensor(np.transpose(image_rgb, (2, 0, 1)), dtype=torch.float32) / 255.0

def scale_to_width_keep_aspect(image_bgr: np.ndarray, width: int) -> np.ndarray:
    if width is None:
        return image_bgr
    h, w = image_bgr.shape[:2]
    n_w = int(width)
    n_h = int((h / w) * n_w)
    return cv2.resize(image_bgr, (n_w, n_h), interpolation=cv2.INTER_AREA)

def extract_image_chips(image_tensor: torch.Tensor, boxes: torch.Tensor, padding: int = 0
    ) -> Tuple[List[np.ndarray], List[tuple]]:
    """Returns chips as RGB uint8 arrays + coords used."""
    img = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    H, W, _ = img.shape
    boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)

    chips, coords = [], []
    for (x1, y1, x2, y2) in boxes_np:
        xi1 = max(0, int(np.floor(x1 - padding)))
        yi1 = max(0, int(np.floor(y1 - padding)))
        xi2 = min(W, int(np.ceil(x2 + padding)))
        yi2 = min(H, int(np.ceil(y2 + padding)))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        chip = img[yi1:yi2, xi1:xi2, :].copy()
        chips.append(chip)
        coords.append((xi1, yi1, xi2, yi2))
    return chips, coords

def draw_labeled(image_tensor: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor,
                 scores: torch.Tensor, distances: torch.Tensor, int_cls: dict) -> np.ndarray:
    """Return an RGB uint8 annotated image (no pyplot blocking)."""
    img = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    # draw with OpenCV for speed
    for box, lbl, scr, dist in zip(boxes.cpu().numpy(),
                                   labels.cpu().numpy(),
                                   scores.cpu().numpy(),
                                   distances.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)  # red box
        cls_name = int_cls.get(int(lbl), str(lbl))
        text = f"{cls_name} {scr:.2f} {dist:.2f}"
        # yellow-ish text with filled red box background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (255, 0, 0), -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return img

def bgr2png_bytes(img_bgr: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if success else b""

def rgb2png_bytes(img_rgb: np.ndarray) -> bytes:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return bgr2png_bytes(img_bgr)

def as_base64_png(img_rgb: np.ndarray) -> str:
    return base64.b64encode(rgb2png_bytes(img_rgb)).decode("utf-8")

# ---------------- Model load (mirrors your script choices) ----------------
device = torch.device("cuda" if (torch.cuda.is_available() and not USE_CPU) else "cpu")
num_classes, class_names = read_yaml(YAML_PATH)
cls_int = {c: i + 1 for i, c in enumerate(class_names)}
cls_int['background'] = 0
int_cls = {v: k for k, v in cls_int.items()}

model = estModel(num_classes=num_classes + 1).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- Flask app ----------------
app = Flask(__name__)

def check_basic_auth(req) -> bool:
    if not BASIC_USER or not BASIC_PASS:
        return True  # auth disabled
    auth = req.authorization
    return bool(auth and auth.username == BASIC_USER and auth.password == BASIC_PASS)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    # Basic Auth (optional)
    if not check_basic_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    # Inputs: multipart/form-data with fields: file (image), fov (float)
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename or "input.png")
    prefix = os.path.splitext(filename)[0]

    try:
        fov = float(request.form.get("fov", "0.57"))
    except Exception:
        return jsonify({"error": "invalid fov"}), 400

    try:
        score_thresh = float(request.form.get("threshold", str(DEFAULT_SCORE_THRESHOLD)))
    except Exception:
        score_thresh = DEFAULT_SCORE_THRESHOLD

    # Optional resize to match your training width (same var name you use)
    try:
        image_size = int(request.form.get("image_size", "1920"))
    except Exception:
        image_size = 1920

    padding = int(request.form.get("padding", "4"))
    return_images_b64 = request.form.get("return_images_b64", "0") == "1"

    # Read file -> BGR numpy
    in_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(in_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    # Resize (keep aspect) like your script (width=image_size)
    image_bgr = scale_to_width_keep_aspect(image_bgr, image_size)
    image_tensor = to_tensor_chw_uint(image_bgr)

    # Build inputs (same dict as your code)
    inputs = {"image": image_tensor.to(device), "fov": torch.tensor(float(fov), dtype=torch.float32).to(device)}

    # Inference
    with torch.no_grad():
        output = model([inputs])[0]

    # Collect tensors on CPU
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()
    distances = output["distances"].cpu()

    # Threshold filter
    mask = scores > score_thresh
    boxes_f = boxes[mask]
    labels_f = labels[mask]
    scores_f = scores[mask]
    distances_f = distances[mask]

    # Extract chips and save them
    chips_rgb, chip_coords = extract_image_chips(image_tensor, boxes_f, padding=padding)
    saved_chips = []
    for i, chip in enumerate(chips_rgb):
        chip_name = f"{prefix}_chip_{i:03d}.png"
        chip_path = os.path.join(OUT_DIR, chip_name)
        cv2.imwrite(chip_path, cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
        saved_chips.append({"file": chip_name, "path": chip_path})

    # Save original + labeled images with original name prefix
    original_name = f"{prefix}_original.png"
    labeled_name  = f"{prefix}_labeled.png"
    original_path = os.path.join(OUT_DIR, original_name)
    labeled_path  = os.path.join(OUT_DIR, labeled_name)
    cv2.imwrite(original_path, image_bgr)

    labeled_rgb = draw_labeled(image_tensor, boxes_f, labels_f, scores_f, distances_f, int_cls)
    cv2.imwrite(labeled_path, cv2.cvtColor(labeled_rgb, cv2.COLOR_RGB2BGR))

    # Build JSON response
    detections = []
    for i, (box, lbl, scr, dist) in enumerate(zip(boxes_f.numpy(),
                                                  labels_f.numpy(),
                                                  scores_f.numpy(),
                                                  distances_f.numpy())):
        detections.append({
            "box": [float(x) for x in box.tolist()],
            "label_id": int(lbl),
            "label": int_cls.get(int(lbl), str(lbl)),
            "score": float(scr),
            "distance": float(dist),
            "chip_file": saved_chips[i]["file"] if i < len(saved_chips) else None
        })

    resp = {
        "prefix": prefix,
        "fov": fov,
        "threshold": score_thresh,
        "count": len(detections),
        "detections": detections,
        "outputs": {
            "original": original_name,
            "labeled": labeled_name,
            "chips": [c["file"] for c in saved_chips]
        }
    }

    # Optionally include images as base64
    if return_images_b64:
        resp["images_b64"] = {
            "original_png": base64.b64encode(bgr2png_bytes(image_bgr)).decode("utf-8"),
            "labeled_png":  as_base64_png(labeled_rgb),
            "chips_png":    [as_base64_png(ch) for ch in chips_rgb]
        }

    return jsonify(resp)

# Optional: serve saved files back (simple static server)
@app.route("/files/<path:fname>", methods=["GET"])
def get_file(fname):
    path = os.path.join(OUT_DIR, fname)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")

# ---------------- CLI entrypoints ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vessel detector API / Gradio UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio UI instead of Flask")
    args = parser.parse_args()

    if not args.gradio:
        # Flask
        app.run(host=args.host, port=args.port, debug=False)
    else:
        # -------- Gradio UI (quick manual testing) --------
        import gradio as gr
        import pandas as pd

        def gr_infer(image: np.ndarray, fov: float, threshold: float, padding: int, width: int):
            # image is RGB (H,W,3) float/uint8
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 255)).astype(np.uint8)

            # Convert to BGR for resizing to match script style
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = scale_to_width_keep_aspect(image_bgr, width)
            image_tensor = to_tensor_chw_uint(image_bgr)

            inputs = {"image": image_tensor.to(device),
                      "fov": torch.tensor(float(fov), dtype=torch.float32).to(device)}
            with torch.no_grad():
                out = model([inputs])[0]

            boxes = out["boxes"].cpu()
            labels = out["labels"].cpu()
            scores = out["scores"].cpu()
            distances = out["distances"].cpu()

            mask = scores > threshold
            boxes_f = boxes[mask]
            labels_f = labels[mask]
            scores_f = scores[mask]
            distances_f = distances[mask]

            # Chips
            chips_rgb, _ = extract_image_chips(image_tensor, boxes_f, padding=padding)
            gallery = [Image.fromarray(ch) for ch in chips_rgb]

            # Labeled
            labeled_rgb = draw_labeled(image_tensor, boxes_f, labels_f, scores_f, distances_f, int_cls)

            # Table
            rows = []
            for box, lbl, scr, dist in zip(boxes_f.numpy(),
                                           labels_f.numpy(),
                                           scores_f.numpy(),
                                           distances_f.numpy()):
                rows.append({
                    "label_id": int(lbl),
                    "label": int_cls.get(int(lbl), str(lbl)),
                    "score": float(scr),
                    "distance": float(dist),
                    "box": [float(x) for x in box.tolist()],
                })
            df = pd.DataFrame(rows)
            return gallery, Image.fromarray(labeled_rgb), df

        with gr.Blocks(title="Vessel Distance/Type Inference") as demo:
            gr.Markdown("## Vessel Detector — Chips, Labels, Distances")
            with gr.Row():
                inp_img = gr.Image(label="Input image", type="numpy")
                with gr.Column():
                    inp_fov = gr.Number(value=0.57, label="FOV")
                    inp_thresh = gr.Slider(0.0, 1.0, value=DEFAULT_SCORE_THRESHOLD, step=0.01, label="Score threshold")
                    inp_pad = gr.Slider(0, 40, value=4, step=1, label="Chip padding (px)")
                    inp_w = gr.Number(value=1920, label="Resize width")
                    btn = gr.Button("Run")

            out_gallery = gr.Gallery(label="Image Chips (RGB)", columns=4, height=300)
            out_labeled = gr.Image(label="Labeled Image")
            out_table = gr.Dataframe(label="Detections (label, score, distance, box)")

            btn.click(fn=gr_infer,
                      inputs=[inp_img, inp_fov, inp_thresh, inp_pad, inp_w],
                      outputs=[out_gallery, out_labeled, out_table])

        demo.launch(server_name=args.host, server_port=args.port, show_api=False)
