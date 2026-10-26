"""Image matching and LabelMe helpers shared by the two dataset commands."""

import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw


DATASETS = {
    "sugar_beet": {"archive": "Sugar_Beet.zip", "folder": "Sugar_Beet",
                   "drive_id": "1kSRJoaP5gFiD-jjRfZHhRgx7GK3lKL3G", "labels": "labels_sb_json.zip"},
    "maize": {"archive": "Maize.zip", "folder": "Maize",
              "drive_id": "1KFQruSa8Xt5T_x7E7H96FTZ1daJJcn0x", "labels": "labels_maize_json.zip"},
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalized(path):
    return str(path).replace("\\", "/")


class ImageSource:
    """Read images in place, from an extracted directory or directly from ZIP."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = None
        if self.path.is_dir():
            self.names = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            self.zip = zipfile.ZipFile(self.path)
            self.names = [n for n in self.zip.namelist()
                          if PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES
                          and "__MACOSX" not in PurePosixPath(n).parts]
        self.by_name = defaultdict(list)
        for name in self.names:
            self.by_name[PurePosixPath(name).name].append(name)
        if not self.names:
            self.close()
            raise ValueError(f"No images in {self.path}")

    def close(self):
        if self.zip:
            self.zip.close()

    def match(self, image_path, split=None, category=None):
        path = normalized(image_path)
        candidates = self.by_name.get(PurePosixPath(path).name, [])
        if split:
            candidates = [n for n in candidates
                          if split.lower() in [p.lower() for p in PurePosixPath(n).parts]]
        if category:
            candidates = [n for n in candidates if category in PurePosixPath(n).parts[:-1]]
        # Prefer the longest matching suffix, never pair by archive/list order.
        scored = []
        parts = PurePosixPath(path).parts
        for name in candidates:
            score = 0
            for a, b in zip(reversed(parts), reversed(PurePosixPath(name).parts)):
                if a.lower() != b.lower():
                    break
                score += 1
            scored.append((score, name))
        best = max((score for score, name in scored), default=0)
        matches = [name for score, name in scored if score == best]
        if len(matches) != 1:
            raise ValueError(f"Expected one image for {image_path!r}, found {len(matches)} in {self.path}")
        return matches[0]

    def open(self, name):
        if self.zip:
            return Image.open(io.BytesIO(self.zip.read(name)))
        return Image.open(self.path / name)


def x_at_y(points, y):
    """Interpolate, or extrapolate the nearest end segment, at a scanline."""
    segments = [(a, b) for a, b in zip(points, points[1:]) if abs(b[1] - a[1]) > 1e-9]
    if not segments:
        raise ValueError("Cannot determine side of a horizontal crop row")
    a, b = min(segments, key=lambda s: max(min(s[0][1], s[1][1]) - y,
                                         y - max(s[0][1], s[1][1]), 0))
    return a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])


def row_names(rows, height):
    """Keep source order; derive side/rank relative to the first (center) row."""
    # Use the lowest scanline shared by all polylines. Extrapolate only when
    # there is no common span. This also handles curved, multi-point rows.
    top = max(min(p[1] for p in row) for row in rows)
    bottom = min(max(p[1] for p in row) for row in rows)
    y = bottom if bottom >= top else height - 1
    positions = [x_at_y(row, y) for row in rows]
    names = ["center_row"] + [None] * (len(rows) - 1)
    for side, indices in (
        ("left", sorted((i for i in range(1, len(rows)) if positions[i] < positions[0]),
                        key=lambda i: positions[i], reverse=True)),
        ("right", sorted((i for i in range(1, len(rows)) if positions[i] >= positions[0]),
                         key=lambda i: positions[i])),
    ):
        for rank, index in enumerate(indices, 1):
            names[index] = f"{side}_row_{rank}"
    return names


def annotation(image_name, size, rows):
    width, height = size
    names = row_names(rows, height)
    return {
        "version": "6.3.1", "flags": {},
        "shapes": [
            {"label": name, "points": points, "group_id": index,
             "description": f"Original MATLAB crop row {index} (1-based order)",
             "shape_type": "line" if len(points) == 2 else "linestrip",
             "flags": {}, "mask": None}
            for index, (name, points) in enumerate(zip(names, rows), 1)
        ],
        "imagePath": PurePosixPath(image_name).name, "imageData": None,
        "imageHeight": height, "imageWidth": width,
    }


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def render_mask(document, original_size, output_size, line_width):
    """Render directly at the output size so line width is in output pixels."""
    width, height = original_size
    if (document.get("imageWidth"), document.get("imageHeight")) != (width, height):
        raise ValueError("JSON dimensions differ from the matched original image")
    mask = Image.new("L", output_size, 0)
    draw = ImageDraw.Draw(mask)
    sx, sy = output_size[0] / width, output_size[1] / height
    for shape in document["shapes"]:
        if shape["shape_type"] not in {"line", "linestrip"}:
            raise ValueError(f"Unsupported row shape: {shape['shape_type']}")
        points = shape["points"]
        if len(points) < 2 or (shape["shape_type"] == "line" and len(points) != 2):
            raise ValueError("Invalid row point count")
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError("Invalid row coordinates")
        draw.line([(round(x * sx), round(y * sy)) for x, y in points],
                  fill=255, width=line_width, joint="curve")
    return mask
