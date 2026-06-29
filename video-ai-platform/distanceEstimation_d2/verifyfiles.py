from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# ----- Config -----
DATASET_ROOT = r"/path/to/dataset"  # contains train/ and valid/ (or val/)
SPLITS = ["train", "valid"]         # will fall back to "val" if "valid" missing
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXT = ".xml"
RECURSIVE = False                    # set True if you have nested subfolders
NORMALIZE_CASE = True                # handle case-insensitive matching
CHECK_WELL_FORMED_XML = False        # set True to parse/validate XML well-formedness
# ------------------

def list_files(dir_path: Path, exts: set[Path]) -> list[Path]:
    if not dir_path.exists():
        return []
    if RECURSIVE:
        return [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

def key_for(p: Path) -> str:
    stem = p.stem
    return stem.lower() if NORMALIZE_CASE else stem

def verify_split(root: Path, split_name: str):
    base = root / split_name
    actual_split = split_name
    if not base.exists() and split_name == "valid":
        # fallback to "val"
        base = root / "val"
        actual_split = "val"

    if not base.exists():
        print(f"[{split_name}] split not found under {root}")
        return None

    images_dir = base / "images"
    labels_dir = base / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[{actual_split}] expected 'images' and 'labels' under {base}")
        return None

    imgs = list_files(images_dir, IMAGE_EXTS)
    lbls = list_files(labels_dir, {LABEL_EXT})

    img_map = {key_for(p): p for p in imgs}
    lbl_map = {}
    for p in lbls:
        k = key_for(p)
        if k not in lbl_map:  # keep first if duplicates
            lbl_map[k] = p

    missing_labels = []
    malformed = []

    for k, img_path in img_map.items():
        lbl_path = lbl_map.get(k)
        if not lbl_path:
            missing_labels.append(img_path)
        elif CHECK_WELL_FORMED_XML:
            try:
                ET.parse(lbl_path)
            except Exception as e:
                malformed.append((lbl_path, str(e)))

    orphan_labels = [p for k, p in lbl_map.items() if k not in img_map]

    print(f"\n=== Split: {actual_split} ===")
    print(f"Images: {len(imgs)} | Labels: {len(lbls)}")
    print(f"Missing labels for images: {len(missing_labels)}")
    for p in missing_labels[:25]:
        print(f"  - {p}")
    if len(missing_labels) > 25:
        print(f"  ... and {len(missing_labels)-25} more")

    if CHECK_WELL_FORMED_XML:
        print(f"Malformed XML files: {len(malformed)}")
        for p, err in malformed[:10]:
            print(f"  - {p} :: {err}")

    print(f"Orphan label files (no image): {len(orphan_labels)}")
    for p in orphan_labels[:15]:
        print(f"  - {p}")
    if len(orphan_labels) > 15:
        print(f"  ... and {len(orphan_labels)-15} more")

    return {
        "split": actual_split,
        "images": len(imgs),
        "labels": len(lbls),
        "missing": len(missing_labels),
        "orphans": len(orphan_labels),
        "malformed": len(malformed),
    }

def main(root_dir: str | None = None):
    root = Path(root_dir or DATASET_ROOT)
    results = []
    for s in SPLITS:
        r = verify_split(root, s)
        if r:
            results.append(r)
    if results:
        print("\n=== Overall ===")
        print(f"Total missing labels: {sum(r['missing'] for r in results)}")
        print(f"Total orphan labels : {sum(r['orphans'] for r in results)}")
        if CHECK_WELL_FORMED_XML:
            print(f"Total malformed XML : {sum(r['malformed'] for r in results)}")

if __name__ == "__main__":
    # Usage:
    #   python verify_pairs.py /data/dataset_root
    # or set DATASET_ROOT above.
    arg_root = "dataset"
    main(arg_root)
