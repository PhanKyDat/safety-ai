"""Copy manually approved pure-background images into PPE train only.

Input folder: data/hard_negative_review/approved_empty/
Only use images verified by a human to contain no target-class instances.
Images with people, helmets, bare heads, vests, or missing-vest cases must be
fully annotated and must NOT be added by this empty-label helper.
Dry-run is the default; pass --apply to modify train/images and train/labels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=r"D:\safety-ai")
    ap.add_argument("--apply", action="store_true", help="Actually copy files; otherwise dry-run")
    args = ap.parse_args()
    root = Path(args.root)
    approved = root / "data" / "hard_negative_review" / "approved_empty"
    dataset = root / "data" / "ppe_v1_round2"
    if not approved.is_dir():
        raise SystemExit(f"Missing review folder: {approved}")
    for split in SPLITS:
        if not (dataset / split / "images").is_dir() or not (dataset / split / "labels").is_dir():
            raise SystemExit(f"Missing expected images/labels for split '{split}' under {dataset}")

    candidates = sorted(p for p in approved.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not candidates:
        print(f"No approved images found in {approved}")
        return 0

    hashes = {}
    for split in SPLITS:
        for p in (dataset / split / "images").iterdir():
            if p.is_file() and p.suffix.lower() in EXTS:
                h = sha256(p)
                hashes.setdefault(h, []).append(f"{split}/{p.name}")

    copied = 0
    skipped = 0
    planned = []
    manifest_rows = []
    train_img = dataset / "train" / "images"
    train_lbl = dataset / "train" / "labels"
    for src in candidates:
        digest = sha256(src)
        if digest in hashes:
            print(f"SKIP duplicate content: {src.name} already in {hashes[digest]}")
            skipped += 1
            manifest_rows.append({"source": str(src), "sha256": digest, "destination": "", "status": "skipped_duplicate"})
            continue
        safe_name = f"hn_{src.stem}_{digest[:8]}{src.suffix.lower()}"
        dest_img = train_img / safe_name
        dest_lbl = train_lbl / f"{Path(safe_name).stem}.txt"
        if dest_img.exists() or dest_lbl.exists():
            print(f"SKIP filename collision: {safe_name}")
            skipped += 1
            manifest_rows.append({"source": str(src), "sha256": digest, "destination": safe_name, "status": "skipped_collision"})
            continue
        planned.append((src, dest_img, dest_lbl, digest))
        manifest_rows.append({"source": str(src), "sha256": digest, "destination": safe_name, "status": "planned_dry_run" if not args.apply else "pending_copy"})
        print(f"{'COPY' if args.apply else 'DRY-RUN'} {src.name} -> train/{safe_name} (empty label)")

    if args.apply:
        for src, dest_img, dest_lbl, digest in planned:
            shutil.copy2(src, dest_img)
            dest_lbl.write_text("", encoding="utf-8")
            copied += 1
        for row in manifest_rows:
            if row["status"] == "pending_copy":
                row["status"] = "copied_to_train_empty_label"
        print(f"\nAdded {copied} verified background images to TRAIN only; skipped {skipped}.")
        print("No val/test files were modified. Keep a manifest of source attribution/licenses.")
    else:
        print(f"\nDry-run only: {len(planned)} would be copied; {skipped} skipped.")
        print("After visual verification, rerun with --apply to make changes.")
    manifest_path = root / "data" / "hard_negative_review" / "import_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "sha256", "destination", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest: {manifest_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
