import os
import sys
import xml.etree.ElementTree as ET
from typing import Tuple, List

# ------- Config -------
ROOT_DIR = r"C:\path\to\your\annotations"  # <-- change this
EXTENSIONS = {".vml", ".xml"}              # validate both .vml and .xml
REQUIRE_FOV_DISTANCE = False               # set True if fov/distance must exist per object
# ----------------------

def read_lines_safe(path: str) -> List[str]:
    # Read as text, be tolerant to encoding issues
    with open(path, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback: replace undecodable bytes
        text = data.decode("utf-8", errors="replace")
    # Normalize CRLF just in case
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

def snippet(lines: List[str], err_line: int, context: int = 2) -> str:
    i = max(1, err_line - context)
    j = min(len(lines), err_line + context)
    out = []
    for ln in range(i, j + 1):
        prefix = ">> " if ln == err_line else "   "
        out.append(f"{prefix}{ln:5d}: {lines[ln-1]}")
    return "\n".join(out)

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def validate_semantics(root: ET.Element, path: str) -> List[str]:
    """
    Basic sanity checks for Pascal VOC-like structure:
      - <filename> exists
      - at least one <object>/<name>
      - optional: each object has numeric <fov> and <distance> if REQUIRE_FOV_DISTANCE=True
    """
    errs = []

    fn = root.find("filename")
    if fn is None or (fn.text or "").strip() == "":
        errs.append("Missing or empty <filename>.")

    objects = root.findall("object")
    if not objects:
        errs.append("No <object> elements found.")
        return errs

    has_name = False
    for idx, obj in enumerate(objects, 1):
        name_el = obj.find("name")
        if name_el is None or (name_el.text or "").strip() == "":
            errs.append(f"Object #{idx}: Missing or empty <name>.")
        else:
            has_name = True

        if REQUIRE_FOV_DISTANCE:
            fov_el = obj.find("fov")
            dist_el = obj.find("distance")
            if fov_el is None or (fov_el.text or "").strip() == "":
                errs.append(f"Object #{idx}: Missing <fov>.")
            elif not is_number(fov_el.text.strip()):
                errs.append(f"Object #{idx}: Non-numeric <fov>: {fov_el.text!r}")

            if dist_el is None or (dist_el.text or "").strip() == "":
                errs.append(f"Object #{idx}: Missing <distance>.")
            elif not is_number(dist_el.text.strip()):
                errs.append(f"Object #{idx}: Non-numeric <distance>: {dist_el.text!r}")

    if not has_name:
        errs.append("No valid <name> found under any <object>.")
    return errs

def validate_file(path: str) -> Tuple[bool, List[str]]:
    """
    Returns (is_valid, errors).
    First checks well-formedness; then runs semantic checks.
    """
    lines = read_lines_safe(path)
    try:
        tree = ET.ElementTree(ET.fromstring("\n".join(lines)))
    except ET.ParseError as e:
        # Extract line & column if available
        msg = str(e)
        if hasattr(e, 'position') and e.position:
            line, col = e.position
            msg += f" (line {line}, column {col})"
            msg += "\nContext:\n" + snippet(lines, line)
        return False, [f"XML not well-formed: {msg}"]

    # Semantic checks
    root = tree.getroot()
    errors = validate_semantics(root, path)
    if errors:
        return False, errors
    return True, []

def walk_files(root_dir: str, exts: set) -> List[str]:
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(r, f))
    return paths

def main():
    #root_dir = ROOT_DIR if len(sys.argv) == 1 else "dataset/valid/labels/"
    root_dir = "dataset/train/labels/"
    files = walk_files(root_dir, EXTENSIONS)
    if not files:
        print(f"No {EXTENSIONS} files found in: {root_dir}")
        return

    total = len(files)
    ok = 0
    failed = 0

    print(f"Validating {total} files under: {root_dir}\n")

    for p in files:
        valid, errs = validate_file(p)
        if valid:
            ok += 1
        else:
            failed += 1
            print(f"[FAIL] {p}")
            for e in errs:
                print(f"  - {e}")
            print("-" * 80)

    print("\n===== SUMMARY =====")
    print(f"Total files : {total}")
    print(f"Valid       : {ok}")
    print(f"Invalid     : {failed}")
    if failed == 0:
        print("All good ✅")
    else:
        print("Some files are malformed or fail semantic checks ❗")

if __name__ == "__main__":
    main()
