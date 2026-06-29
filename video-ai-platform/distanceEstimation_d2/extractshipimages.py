import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    # Convert CHW [0,1] -> HWC uint8 (RGB)
    img = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    H, W, _ = img.shape

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    chips, coords = [], []
    for (x1, y1, x2, y2) in boxes_np:
        # Pad and clamp
        xi1 = max(0, int(np.floor(x1 - padding)))
        yi1 = max(0, int(np.floor(y1 - padding)))
        xi2 = min(W, int(np.ceil(x2 + padding)))
        yi2 = min(H, int(np.ceil(y2 + padding)))

        if xi2 <= xi1 or yi2 <= yi1:  # skip invalid
            continue

        chip = img[yi1:yi2, xi1:xi2, :].copy()
        chips.append(chip)
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
        threshold (float): optional score filter applied here (in addition to any prior filter).
        dpi (int): resolution for labeled figure.

    Returns:
        original_path (str), labeled_path (str)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare image arrays
    img_rgb = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # (Optional) filter by threshold
    if threshold is not None:
        mask = scores > threshold
        boxes_ = boxes[mask]
        labels_ = labels[mask]
        scores_ = scores[mask]
        distances_ = distances[mask]
    else:
        boxes_, labels_, scores_, distances_ = boxes, labels, scores, distances

    # Save original
    original_path = os.path.join(out_dir, f"{base_name}_original.png")
    cv2.imwrite(original_path, img_bgr)

    # Draw labeled with matplotlib (RGB)
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
