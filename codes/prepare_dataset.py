#!/usr/bin/env python3
"""Download CRDLD sugar beet/maize RGB images and render their row masks.

Only Pillow is required. Downloads and ZIP operations use the standard library.
Image ZIPs are temporary by default; JSON definitions are read from labels/*.zip.
"""

import argparse
import http.cookiejar
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from PIL import Image

from dataset_labels import DATASETS, ImageSource, json_bytes, render_mask


ROOT = Path(__file__).resolve().parents[1]
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1QrL0-bGUCSY1T2mACdZKVKDcCV8zIeDj?usp=sharing"


class DownloadForm(HTMLParser):
    """Parse Google's large-file confirmation form without extra dependencies."""

    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.active = True
        elif self.active and tag == "input" and "name" in attrs:
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def download_google_zip(file_id, destination):
    """Stream a public Drive ZIP, including the large-file confirmation step."""
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "CRDLD-dataset-preparation/1.0")]
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": file_id, "export": "download"})
    pending = destination.with_suffix(".zip.part")
    try:
        for attempt in range(3):
            with opener.open(url, timeout=120) as response:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    page = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
                    form = DownloadForm()
                    form.feed(page)
                    if not form.action:
                        raise ValueError("Google Drive did not return a ZIP or a download form. "
                                         "Check public sharing, download quota, or use --archive-dir.")
                    action = urllib.parse.urljoin(url, form.action)
                    parsed = urllib.parse.urlsplit(action)
                    if parsed.scheme != "https" or parsed.hostname not in {
                            "drive.google.com", "drive.usercontent.google.com"}:
                        raise ValueError("Unexpected Google download confirmation host")
                    separator = "&" if parsed.query else "?"
                    url = action + separator + urllib.parse.urlencode(form.fields)
                    continue
                expected = response.headers.get("Content-Length")
                with pending.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if expected and pending.stat().st_size != int(expected):
                    raise ValueError("Incomplete Google Drive download")
            if not zipfile.is_zipfile(pending):
                raise ValueError("Google Drive response is not a ZIP archive")
            pending.replace(destination)
            return
        raise ValueError("Google Drive confirmation did not produce a download")
    finally:
        pending.unlink(missing_ok=True)


def prepare(images, labels_path, output, height=None, width=None, line_width=6,
            image_format="png", overwrite=False):
    """Match and validate the full dataset before writing any image pairs."""
    jobs = []
    targets = set()
    seen_images = set()
    with zipfile.ZipFile(labels_path) as labels:
        for member in sorted(labels.namelist()):
            path = PurePosixPath(member)
            if path.suffix.lower() != ".json":
                continue
            if len(path.parts) != 3 or path.parts[0] not in {"train", "test", "validation"}:
                raise ValueError(f"Expected split/category/filename.json, got {member}")
            split, category, filename = path.parts
            document = json.loads(labels.read(member))
            name = images.match(document["imagePath"], split, category)
            if name in seen_images:
                raise ValueError(f"Multiple labels for image {name}")
            seen_images.add(name)
            output_name = PurePosixPath(name).stem + "." + image_format
            image_target = output / split / "image" / output_name
            label_target = output / split / "label" / output_name
            if image_target in targets:
                raise ValueError(f"Duplicate output filename {output_name} in {split}")
            targets.add(image_target)
            for target in (image_target, label_target):
                if target.exists() and not overwrite:
                    raise ValueError(f"Output exists: {target}; use --overwrite or another --output")
            with images.open(name) as image:
                original_size = image.size
            if (document.get("imageWidth"), document.get("imageHeight")) != original_size:
                raise ValueError(f"Image size mismatch: {member}")
            output_size = (width, height) if width else original_size
            # Validate coordinates without allocating a full-size mask in preflight.
            render_mask(document, original_size, (1, 1), 1)
            jobs.append((name, document, original_size, output_size, image_target, label_target))
    if not jobs:
        raise ValueError(f"No image annotations found in {labels_path}")
    counts = {}
    for index, (name, document, original_size, size, image_target, label_target) in enumerate(jobs, 1):
        with images.open(name) as original:
            image = original.convert("RGB")
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
        mask = render_mask(document, original_size, size, line_width)
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        options = {"quality": 100, "subsampling": 0} if image_format == "jpg" else {}
        image.save(image_target, **options)
        mask.save(label_target, **options)
        split = image_target.parent.parent.name
        counts[split] = counts.get(split, 0) + 1
        if index % 100 == 0 or index == len(jobs):
            print(f"  Prepared {index}/{len(jobs)} image/mask pairs", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preparation.json").write_bytes(json_bytes({
        "source": str(images.path), "labels": str(labels_path), "images": counts,
        "height": height, "width": width, "line_width": line_width,
        "format": image_format, "mask_background": 0, "mask_row": 255,
    }))
    return counts


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images-root", type=Path,
                        help="Use extracted originals instead of downloading")
    source.add_argument("--archive-dir", type=Path,
                        help="Use existing Sugar_Beet.zip / Maize.zip instead of downloading")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    parser.add_argument("--output", type=Path, default=ROOT / "data",
                        help="Output root; each crop gets train/test image/label directories")
    parser.add_argument("--height", type=positive, help="Output height (requires --width)")
    parser.add_argument("--width", type=positive, help="Output width (requires --height)")
    parser.add_argument("--line-width", type=positive, default=6,
                        help="White row width in output pixels (default: 6)")
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        help="PNG keeps masks binary; JPG uses lossy compression")
    parser.add_argument("--keep-downloads", type=Path, metavar="DIRECTORY",
                        help="Retain downloaded ZIPs here; otherwise remove temporary downloads")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing image/mask pairs")
    args = parser.parse_args()
    if bool(args.height) != bool(args.width):
        parser.error("--height and --width must be supplied together")
    if args.keep_downloads and (args.images_root or args.archive_dir):
        parser.error("--keep-downloads applies only to downloads")
    selected = DATASETS if args.dataset == "both" else [args.dataset]
    for key in selected:
        labels = args.labels_dir / DATASETS[key]["labels"]
        if not labels.is_file():
            parser.error(f"Missing annotation ZIP: {labels}")
    with tempfile.TemporaryDirectory(prefix="crdld-") as temporary:
        for key in selected:
            dataset = DATASETS[key]
            print(f"Preparing {key}", flush=True)
            if args.images_root:
                path = args.images_root / dataset["folder"]
                if not path.is_dir():
                    path = args.images_root
            elif args.archive_dir:
                path = args.archive_dir / dataset["archive"]
            else:
                download_dir = args.keep_downloads or Path(temporary)
                download_dir.mkdir(parents=True, exist_ok=True)
                path = download_dir / dataset["archive"]
                if not path.is_file():
                    print(f"  Downloading {dataset['archive']} from Google Drive...", flush=True)
                    download_google_zip(dataset["drive_id"], path)
            images = ImageSource(path)
            try:
                counts = prepare(images, args.labels_dir / dataset["labels"], args.output / key,
                                 args.height, args.width, args.line_width, args.format, args.overwrite)
            finally:
                images.close()
            print(f"{key}: {counts} -> {args.output / key}", flush=True)
            if not args.images_root and not args.archive_dir and not args.keep_downloads:
                path.unlink()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Preparation failed: {error}")
