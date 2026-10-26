"""Regression tests for filename matching, coordinate conversion, and masks."""

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
from convert_mat_labels import convert
from dataset_labels import ImageSource, annotation, json_bytes, render_mask, row_names
from matlab_labels import ground_truth
from prepare_dataset import download_google_zip, prepare


def element(tag, payload):
    padding = b"" if tag == 15 else b"\0" * (-len(payload) % 8)
    return struct.pack("<II", tag, len(payload)) + payload + padding


def mat_matrix(kind, dims, body):
    return element(14, element(6, struct.pack("<II", kind, 0))
                   + element(5, struct.pack("<" + "i" * len(dims), *dims))
                   + element(1, b"") + body)


def cell(children):
    return mat_matrix(1, (len(children), 1), b"".join(children))


def mat_struct(records):
    names = list(records[0])
    width = max(map(len, names)) + 1
    body = element(5, struct.pack("<i", width))
    body += element(1, b"".join(n.encode().ljust(width, b"\0") for n in names))
    body += b"".join(record[n] for record in records for n in names)
    return mat_matrix(2, (len(records), 1), body)


def legacy_mat():
    """Small serialized subsystem fixture, including an N>2 curved row."""
    path = r"C:\old\Maize\1\Train\frame.png"
    source = mat_matrix(4, (1, len(path)), element(16, path.encode()))
    def line(points):
        values = [p[0] for p in points] + [p[1] for p in points]
        return mat_matrix(6, (len(points), 2), element(9, struct.pack("<" + "d" * len(values), *values)))
    rows = cell([line([[41, 2], [41, 50]]), line([[10, 2], [12, 25], [15, 50]])])
    gt = mat_struct([{
        "DataSource": mat_struct([{"Source": cell([source])}]),
        "LabelData": mat_struct([{"crop": rows}]),
    }])
    subsystem = mat_matrix(9, (1, len(gt) + 8), element(2, b"\x00\x01IM\0\0\0\0" + gt))
    header = b"MATLAB 5.0 MAT-file".ljust(124, b" ") + b"\x00\x01IM"
    return header + element(15, zlib.compress(subsystem))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.images = self.root / "originals"
        for split, color in [("Train", "red"), ("Test", "blue")]:
            path = self.images / "Maize" / "1" / split / "frame.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (80, 50), color).save(path)
        self.document = annotation("frame.png", (80, 50), [
            [[40, 1], [40, 49]], [[9, 1], [11, 24], [14, 49]],
        ])
        self.labels = self.root / "labels.zip"
        with zipfile.ZipFile(self.labels, "w") as archive:
            archive.writestr("train/1/frame.json", json_bytes(self.document))

    def source(self, path=None):
        images = ImageSource(path or self.images)
        self.addCleanup(images.close)
        return images

    def test_directory_and_zip_match_windows_path_by_split_and_category(self):
        archive = self.root / "Maize.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for path in self.images.rglob("*.png"):
                output.write(path, path.relative_to(self.images).as_posix())
        for images in [self.source(), self.source(archive)]:
            self.assertEqual(images.match(r"C:\old\Maize\1\Train\frame.png", "train", "1"),
                             "Maize/1/Train/frame.png")
            with self.assertRaisesRegex(ValueError, "found 2"):
                images.match("frame.png")
            with self.assertRaisesRegex(ValueError, "found 0"):
                images.match("missing.png", "train", "1")

    def test_original_size_rgb_and_binary_mask(self):
        output = self.root / "prepared"
        with contextlib.redirect_stdout(io.StringIO()):
            counts = prepare(self.source(), self.labels, output)
        self.assertEqual(counts, {"train": 1})
        with Image.open(output / "train/image/frame.png") as rgb:
            self.assertEqual((rgb.size, rgb.mode), ((80, 50), "RGB"))
            self.assertEqual(rgb.getpixel((40, 25)), (255, 0, 0))
        with Image.open(output / "train/label/frame.png") as mask:
            self.assertEqual((mask.size, mask.mode), ((80, 50), "L"))
            self.assertEqual(set(mask.tobytes()), {0, 255})
            self.assertEqual(mask.getpixel((40, 25)), 255)

    def test_resize_scales_xy_and_keeps_requested_pixel_width(self):
        document = annotation("frame.png", (80, 50), [[[40, 0], [40, 49]]])
        mask = render_mask(document, (80, 50), (160, 25), 3)
        bright = [x for x in range(mask.width) if mask.getpixel((x, 12))]
        self.assertEqual(bright, [79, 80, 81])
        self.assertEqual(set(mask.tobytes()), {0, 255})

    def test_curved_row_keeps_all_segments(self):
        document = annotation("frame.png", (80, 50), [[[10, 0], [40, 25], [70, 49]]])
        self.assertEqual(document["shapes"][0]["shape_type"], "linestrip")
        mask = render_mask(document, (80, 50), (80, 50), 1)
        for point in [(10, 0), (40, 25), (70, 49)]:
            self.assertEqual(mask.getpixel(point), 255)

    def test_row_names_preserve_variable_original_order_and_rank_outward(self):
        def row(x):
            return [[x, 0], [x, 49]]
        rows = [row(40), row(65), row(75), row(20), row(5)]
        self.assertEqual(row_names(rows, 50), [
            "center_row", "right_row_1", "right_row_2", "left_row_1", "left_row_2"])
        document = annotation("frame.png", (80, 50), rows)
        self.assertEqual([s["group_id"] for s in document["shapes"]], [1, 2, 3, 4, 5])
        self.assertEqual([s["points"] for s in document["shapes"]], rows)

    def test_matlab_subsystem_coordinates_are_shifted_once(self):
        records = ground_truth(legacy_mat())
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0][0].endswith(r"Train\frame.png"))
        self.assertEqual(records[0][1], [
            [[40, 1], [40, 49]], [[9, 1], [11, 24], [14, 49]]])

    def test_converter_roundtrip_and_failed_conversion_preserves_destination(self):
        legacy = self.root / "legacy.zip"
        destination = self.root / "converted.zip"
        with zipfile.ZipFile(legacy, "w") as archive:
            archive.writestr("Train/1TRN.mat", legacy_mat())
        report = convert(legacy, destination, self.source())
        self.assertEqual(report["total"], 1)
        with zipfile.ZipFile(destination) as archive:
            document = json.loads(archive.read("train/1/frame.json"))
        self.assertEqual(document, self.document)
        old = destination.read_bytes()
        with zipfile.ZipFile(legacy, "w") as archive:
            archive.writestr("Train/2TRN.mat", legacy_mat())
        with self.assertRaisesRegex(ValueError, "found 0"):
            convert(legacy, destination, self.source())
        self.assertEqual(destination.read_bytes(), old)
        self.assertTrue(legacy.exists())

    def test_preflight_rejects_size_mismatch_and_missing_image(self):
        for changes in [{"imageWidth": 81}, {"imagePath": "missing.png"}]:
            with zipfile.ZipFile(self.labels, "w") as archive:
                archive.writestr("train/1/frame.json", json_bytes(dict(self.document, **changes)))
            output = self.root / "prepared"
            with self.assertRaises(ValueError):
                prepare(self.source(), self.labels, output)
            self.assertFalse(output.exists())

    def test_existing_output_requires_explicit_overwrite(self):
        output = self.root / "prepared"
        existing = output / "train/image/frame.png"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "Output exists"):
            prepare(self.source(), self.labels, output)
        self.assertEqual(existing.read_bytes(), b"keep me")

    def test_google_confirmation_and_download_streaming(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("file.txt", "test")
        payload = data.getvalue()
        class Response(io.BytesIO):
            def __init__(self, body, content_type):
                super().__init__(body)
                self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        page = (b'<form id="download-form" action="https://drive.usercontent.google.com/download">'
                b'<input name="id" value="file-id"><input name="confirm" value="t"></form>')
        responses = [Response(page, "text/html"), Response(payload, "application/zip")]
        with patch("prepare_dataset.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = responses
            destination = self.root / "download.zip"
            download_google_zip("file-id", destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertIn("confirm=t", opener.return_value.open.call_args_list[1].args[0])
            self.assertFalse(destination.with_suffix(".zip.part").exists())

    def test_google_html_error_does_not_become_a_zip(self):
        response = io.BytesIO(b"<html>Download quota exceeded</html>")
        response.headers = {"Content-Type": "text/html"}
        destination = self.root / "download.zip"
        with patch("prepare_dataset.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(ValueError, "quota"):
                download_google_zip("file-id", destination)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".zip.part").exists())


if __name__ == "__main__":
    unittest.main()
