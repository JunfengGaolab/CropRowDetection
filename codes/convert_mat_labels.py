#!/usr/bin/env python3
"""Convert the legacy CRDLD .mat ZIPs to per-image LabelMe JSON ZIPs."""

import argparse
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from dataset_labels import DATASETS, ImageSource, annotation, json_bytes
from matlab_labels import ground_truth, label_definition


ROOT = Path(__file__).resolve().parents[1]


def convert(source, destination, images):
    """Validate every record before atomically publishing a new JSON archive."""
    counts = Counter()
    patterns = Counter()
    used_images = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        pending = Path(temporary) / destination.name
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(
                pending, "w", compression=zipfile.ZIP_DEFLATED) as new:
            mat_names = sorted(n for n in old.namelist() if n.lower().endswith(".mat"))
            if not mat_names:
                raise ValueError(f"No MATLAB annotations in {source}")
            for member in mat_names:
                match = re.fullmatch(r"(\d+)(TRN|TST)\.mat", PurePosixPath(member).name, re.I)
                if not match:
                    raise ValueError(f"Unrecognized category/split in {member}")
                category = match[1]
                split = "train" if match[2].upper() == "TRN" else "test"
                for image_path, rows in ground_truth(old.read(member)):
                    name = images.match(image_path, split, category)
                    if name in used_images:
                        raise ValueError(f"Image annotated more than once: {name}")
                    used_images.add(name)
                    with images.open(name) as image:
                        document = annotation(name, image.size, rows)
                    json_name = f"{split}/{category}/{PurePosixPath(name).stem}.json"
                    new.writestr(json_name, json_bytes(document))
                    counts[split] += 1
                    patterns[tuple(s["label"] for s in document["shapes"])] += 1
        with zipfile.ZipFile(pending) as check:
            if check.testzip() is not None or len(check.namelist()) != sum(counts.values()):
                raise ValueError("Converted archive verification failed")
        pending.replace(destination)
    return {"images": dict(sorted(counts.items())), "total": sum(counts.values()),
            "row_order_patterns": [{"order": list(order), "images": count}
                                   for order, count in patterns.most_common()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, type=Path,
                        help="Directory containing extracted Sugar_Beet and Maize folders")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "labels")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--remove-originals", action="store_true",
                        help="Delete legacy ZIPs/labeldef.mat only after all conversions succeed")
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "both" else [args.dataset]
    report = {}
    originals = []
    for key in selected:
        dataset = DATASETS[key]
        source = args.labels_dir / ("labels_sb.zip" if key == "sugar_beet" else "labels_maize.zip")
        root = args.images_root / dataset["folder"]
        images = ImageSource(root if root.is_dir() else args.images_root)
        try:
            report[key] = convert(source, args.output_dir / dataset["labels"], images)
        finally:
            images.close()
        originals.append(source)
        print(f"{key}: {report[key]['total']} annotations -> {args.output_dir / dataset['labels']}")
    definition = args.labels_dir / "labeldef.mat"
    if definition.exists():
        (args.output_dir / "labeldef.json").write_bytes(json_bytes(label_definition(definition.read_bytes())))
        originals.append(definition)
    (args.output_dir / "conversion_report.json").write_bytes(json_bytes(report))
    if args.remove_originals:
        for source in originals:
            source.unlink()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Conversion failed: {error}")
