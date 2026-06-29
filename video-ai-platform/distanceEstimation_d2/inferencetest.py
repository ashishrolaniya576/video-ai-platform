import os
import yaml 
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from extractshipimages import   extract_image_chips,save_original_and_labeled
from model_utils import estModel


model_PATH = "best_100epocs.pth"
yaml_PATH = "data.yaml" # Use same xml as used for training
image_PATH = "dataset/train/images/Screenshot from 2025-09-08 06-39-06.png"
fov = 0.57 # FOV value of this image
image_size  = 1920 # Ensure this value is same as used in training
threshold = 0.3 # show objects with > 70% confidence scores


def read_YAML(path):
    assert os.path.exists(path), f"File doesn't exist: {path}"
    # Read data.yaml
    with open(path, 'r') as file:
        contents = yaml.safe_load(file)

    return contents['nc'], contents['names']

def scale_IMG(path, size=None):
    image = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB

    if (size):
        h, w, _ = image.shape
        aspect_ratio = h / w
        n_w = int(size)
        n_h = int(aspect_ratio * n_w)
                    
        image = cv2.resize(image, (n_w, n_h), interpolation=cv2.INTER_AREA)

    return torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255.0 



# Read YAML file to get 
num_classes, class_names  = read_YAML(yaml_PATH)

# Read an image
image = scale_IMG(image_PATH, image_size)
# Create input dictionary with image and fov
inputs = {'image': image, 'fov': torch.tensor(fov, dtype=torch.float32)}

# create class to int and vice versa dictionary [to be used for visualization]
cls_int = {c: i+1 for i, c in enumerate(class_names)}
cls_int['background'] = 0
int_cls = { values:keys for keys, values in cls_int.items()}

# Init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = estModel(num_classes=num_classes + 1).to(device)
#model.load_state_dict(torch.load(model_PATH))
model.load_state_dict(torch.load(model_PATH, map_location=torch.device('cpu')))


# Inference on single image
model.eval()
with torch.no_grad():
    output = model([{k: v.to(device) for k, v in inputs.items()}])[0]

# Filter predictions by score threshold
boxes = output['boxes'].cpu()
labels = output['labels'].cpu()
scores = output['scores'].cpu()
distances = output['distances'].cpu()

print(labels,scores,distances)

selected_indices = scores > threshold
boxes = boxes[selected_indices]
labels = labels[selected_indices]
scores = scores[selected_indices]
distances = distances[selected_indices]
print(boxes, labels, scores, distances)



# Visualize on a single image
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(np.transpose(image, (1, 2, 0)))

for box, label, score, distance in zip(boxes, labels, scores, distances):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    class_name = int_cls[label.item()]
    ax.text(x1, y1, f"{class_name} {score:.2f} {distance:.2f}", color='yellow',
            fontsize=12, bbox=dict(facecolor='red', alpha=0.6))
    print("classname",class_name)
    print("distance",distance)
    print("score",score)

plt.axis('off')
plt.show()




# Function 1: extract image chips for each bounding box
# 
def extract_image_chips(image_tensor, boxes, padding: int = 0):
    """
    Extract RGB image chips for each bounding box.

    Args:
        image_tensor (torch.Tensor): CHW float32 tensor in [0,1].
        boxes (torch.Tensor): Nx4 tensor [x1, y1, x2, y2].
        padding (int): extra pixels to include around each box (clamped to image).

    Returns:
        chips (List[np.ndarray]): list of HxWx3 RGB uint8 arrays.
        coords (List[tuple]): list of (x1, y1, x2, y2) used after padding/clamping.
    """
    img = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    H, W, _ = img.shape

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

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


# Function 2: save original and labeled image
def save_original_and_labeled(
    image_tensor,
    boxes,
    labels,
    scores,
    distances,
    int_cls: dict,
    out_dir: str,
    base_name: str = "result",
    threshold: float = None,
    dpi: int = 150,
):
    """
    Save the original RGB image and a labeled version with boxes/text.

    Args:
        image_tensor (torch.Tensor): CHW float32 tensor in [0,1].
        boxes, labels, scores, distances (torch.Tensors): model outputs.
        int_cls (dict): {int_label: class_name}
        out_dir (str): output directory path.
        base_name (str): filename stem.
        threshold (float): optional score filter applied here.
        dpi (int): resolution for labeled figure.

    Returns:
        original_path (str), labeled_path (str)
    """
    os.makedirs(out_dir, exist_ok=True)

    img_rgb = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if threshold is not None:
        mask = scores > threshold
        boxes_ = boxes[mask]
        labels_ = labels[mask]
        scores_ = scores[mask]
        distances_ = distances[mask]
    else:
        boxes_, labels_, scores_, distances_ = boxes, labels, scores, distances

    original_path = os.path.join(out_dir, f"{base_name}_original.png")
    cv2.imwrite(original_path, img_bgr)

    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=dpi)
    ax.imshow(img_rgb)
    ax.axis('off')

    if len(boxes_) > 0:
        for box, lbl, scr, dist in zip(
            boxes_.cpu().numpy(),
            labels_.cpu().numpy(),
            scores_.cpu().numpy(),
            distances_.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            cls_name = int_cls.get(int(lbl), str(lbl))
            ax.text(
                x1, y1,
                f"{cls_name} {scr:.2f} {dist:.2f}",
                color='yellow',
                fontsize=12,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=2),
            )

    labeled_path = os.path.join(out_dir, f"{base_name}_labeled.png")
    plt.savefig(labeled_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return original_path, labeled_path


def extract_image_chips(image_tensor, boxes, padding: int = 0, prefix: str = "img"):
    """
    Extract RGB image chips for each bounding box, prefixing filenames.

    Args:
        image_tensor (torch.Tensor): CHW float32 tensor in [0,1].
        boxes (torch.Tensor): Nx4 tensor [x1, y1, x2, y2].
        padding (int): extra pixels to include around each box.
        prefix (str): prefix for saving chip filenames.

    Returns:
        chips (List[np.ndarray]): list of HxWx3 RGB uint8 arrays.
        coords (List[tuple]): list of (x1, y1, x2, y2).
    """
    img = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    H, W, _ = img.shape

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    chips, coords = [], []
    for idx, (x1, y1, x2, y2) in enumerate(boxes_np):
        xi1 = max(0, int(np.floor(x1 - padding)))
        yi1 = max(0, int(np.floor(y1 - padding)))
        xi2 = min(W, int(np.ceil(x2 + padding)))
        yi2 = min(H, int(np.ceil(y2 + padding)))

        if xi2 <= xi1 or yi2 <= yi1:
            continue

        chip = img[yi1:yi2, xi1:xi2, :].copy()
        chips.append((chip, f"{prefix}_chip_{idx:03d}.png"))
        coords.append((xi1, yi1, xi2, yi2))

    return chips, coords


def save_original_and_labeled(
    image_tensor,
    boxes,
    labels,
    scores,
    distances,
    int_cls: dict,
    out_dir: str,
    base_name: str,
    threshold: float = None,
    dpi: int = 150,
):
    """
    Save the original RGB image and a labeled version with boxes/text,
    prefixing filenames with base_name (e.g., source image name).
    """
    os.makedirs(out_dir, exist_ok=True)

    img_rgb = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if threshold is not None:
        mask = scores > threshold
        boxes_ = boxes[mask]
        labels_ = labels[mask]
        scores_ = scores[mask]
        distances_ = distances[mask]
    else:
        boxes_, labels_, scores_, distances_ = boxes, labels, scores, distances

    original_path = os.path.join(out_dir, f"{base_name}_original.png")
    cv2.imwrite(original_path, img_bgr)

    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=dpi)
    ax.imshow(img_rgb)
    ax.axis('off')

    if len(boxes_) > 0:
        for box, lbl, scr, dist in zip(
            boxes_.cpu().numpy(),
            labels_.cpu().numpy(),
            scores_.cpu().numpy(),
            distances_.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            cls_name = int_cls.get(int(lbl), str(lbl))
            ax.text(
                x1, y1,
                f"{cls_name} {scr:.2f} {dist:.2f}",
                color='yellow',
                fontsize=12,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=2),
            )

    labeled_path = os.path.join(out_dir, f"{base_name}_labeled.png")
    plt.savefig(labeled_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return original_path, labeled_path


# ------------------------------ USAGE (APPENDED) ------------------------------

# Derive prefix from the original image filename
prefix = os.path.splitext(os.path.basename(image_PATH))[0]

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

# 1) Extract chips
chips, chip_coords = extract_image_chips(image, boxes, padding=4, prefix=prefix)
for chip, fname in chips:
    cv2.imwrite(os.path.join(out_dir, fname), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))

# 2) Save original + labeled images
orig_path, lab_path = save_original_and_labeled(
    image_tensor=image,
    boxes=boxes,
    labels=labels,
    scores=scores,
    distances=distances,
    int_cls=int_cls,
    out_dir=out_dir,
    base_name=prefix,   # prefix matches original image
    threshold=None,
    dpi=150,
)

print(f"Saved: {orig_path}")
print(f"Saved: {lab_path}")
print(f"Saved {len(chips)} chips to: {out_dir}")