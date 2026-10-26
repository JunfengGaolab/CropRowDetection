# CRDLD v3.0 dataset preparation

The public [Google Drive folder](https://drive.google.com/drive/folders/1QrL0-bGUCSY1T2mACdZKVKDcCV8zIeDj?usp=sharing) contains `Sugar_Beet.zip` and `Maize.zip`. The scripts require **Python 3.10+ and Pillow only**, in addition to the standard library. They do not require MATLAB, SciPy, OpenCV, TensorFlow, gdown, or a Google API key.

```bash
python -m pip install -r codes/requirements-data.txt
python codes/prepare_dataset.py
```

Run commands from the repository root. The default command downloads both image archives, handles Google's large-file confirmation form, reads images directly from the ZIPs, and reads the matching JSON annotations from the repository's `labels/` ZIPs. Temporary downloads are removed after processing, including when the command fails. Downloading both crops transfers approximately 3.9 GB; allow disk space for the temporary archive and generated images.

The default output is:

```text
data/
  sugar_beet/
    train/image/<original-image-stem>.png
    train/label/<original-image-stem>.png
    test/image/<original-image-stem>.png
    test/label/<original-image-stem>.png
    preparation.json
  maize/
    train/image/...
    train/label/...
    test/image/...
    test/label/...
    preparation.json
```

RGB images and single-channel masks have identical filenames and dimensions. Masks have black background (0) and white rows (255). PNG is the default because it preserves exact binary mask values. `--format jpg` produces JPEG images and masks, but compression introduces intermediate values at mask edges.

Image sizes are preserved by default: the supplied originals are 1280 × 720. To force a size, supply **both** dimensions:

```bash
python codes/prepare_dataset.py --dataset maize --height 512 --width 512 --line-width 8
```

`--line-width` defaults to 6, following `label_image_gen.m`, and is measured in **output pixels**. Coordinates are scaled independently in x and y when resizing; both the RGB image and mask get the specified dimensions. Masks are drawn directly at their final size so resizing cannot blur the row/background boundary. This replaces MATLAB's figure/screenshot rendering; it does not reproduce figure-dependent linewidth or screenshot artifacts.

Existing outputs cause an error before processing that crop. Use a new `--output` directory or explicitly pass `--overwrite` to replace pairs. `--overwrite` does not remove unrelated or stale files from a prior run. The default output is `data/` at the repository root.

## Local inputs and retained downloads

```bash
# Use extracted originals, avoiding downloads entirely.
# /path/to/originals contains Sugar_Beet/ and Maize/.
python codes/prepare_dataset.py --images-root /path/to/originals

# A selected crop's own directory can also be supplied directly.
python codes/prepare_dataset.py --dataset maize --images-root /path/to/Maize

# Read existing image ZIPs without unpacking or deleting them.
python codes/prepare_dataset.py --archive-dir /path/to/image-zips

# Optional: retain new downloads instead of using temporary storage.
python codes/prepare_dataset.py --keep-downloads /path/to/image-zips
```

`--archive-dir` expects the original archive names, `Sugar_Beet.zip` and `Maize.zip`. `--labels-dir` can select another directory containing `labels_sb_json.zip` and `labels_maize_json.zip`. Local images and local ZIPs are never deleted. Google files must allow public downloads; sharing restrictions or quota errors are reported rather than saved as ZIP files.

Matching uses the **original filename, split, and category**, rather than list order or numeric renaming. Ambiguous or missing matches, duplicate output names, and mismatched image dimensions are errors. Both original Windows paths and ZIP paths are supported. No arbitrary ZIP paths are extracted to disk.

## LabelMe JSON and row order

Each annotation ZIP contains `train/<category>/<original-image-stem>.json` and `test/<category>/<original-image-stem>.json`. Every JSON follows the supplied LabelMe 6.3.1 example: `version`, `flags`, `shapes`, `imagePath`, `imageData: null`, `imageHeight`, and `imageWidth`. A two-point row has `shape_type: "line"`; a curved row with more points has `shape_type: "linestrip"`. All original polyline points are retained. MATLAB's 1-based pixel coordinates are shifted by -1 in x and y for LabelMe/Pillow's 0-based coordinates. Coordinates outside the image are preserved; drawing clips them at the image boundary.

The first row is `center_row`. **There is no single fixed order for the remaining rows.** Across all converted images, the most frequent patterns are:

| Crop | Original row order, interpreted spatially | Images |
| --- | --- | ---: |
| Sugar beet | center, left 1, left 2, right 1, right 2 | 672 / 1,505 |
| Sugar beet | center, left 1, right 1, right 2, left 2 | 260 / 1,505 |
| Maize | center, left 1, right 1, right 2, left 2 | 129 / 245 |
| Maize | center, left 1, left 2, right 1, right 2 | 42 / 245 |

JSON `shapes` stay in the original MATLAB row order. Each shape's `group_id` is its **1-based original row index**, also recorded in its description. Spatial names (`left_row_1`, `left_row_2`, `right_row_1`, etc.) count outward from the center row, from the camera's point of view. They are derived from x positions at the lowest scanline shared by all rows; if there is no common vertical span, the nearest end segments are extrapolated to the image bottom. This preserves source order while making side identity explicit. The complete pattern counts are in `labels/conversion_report.json`. Every row contributes to the binary training mask, regardless of its name.

| Annotation archive | Train | Test | Total |
| --- | ---: | ---: | ---: |
| `labels/labels_sb_json.zip` | 1,075 | 430 | 1,505 |
| `labels/labels_maize_json.zip` | 175 | 70 | 245 |

`labels/labeldef.json` preserves the standalone MATLAB label-definition metadata (the original `crop` label and its color). It is metadata rather than a per-image annotation. The JSON archives contain only per-image JSON files; legacy MATLAB utilities and spreadsheets from the old archives are not copied into them.

To reproduce the conversion from legacy archives:

```bash
python codes/convert_mat_labels.py \
  --images-root /path/to/originals \
  --labels-dir /path/to/legacy-labels \
  --output-dir labels
# Add --remove-originals to delete legacy ZIPs and labeldef.mat after success.
```

The converter reads the specific MATLAB v5 `groundTruth` representation supplied with CRDLD using the standard library. It verifies every image match and gets dimensions from the actual image. It does not support arbitrary MATLAB objects or v7.3/HDF5 files. All requested conversions must succeed before `--remove-originals` removes source files. A ZIP is published only after its records and CRCs pass verification.

## Using the generated dataset

Each crop contains `train/` and `test/` splits with paired `image/` and `label/` directories. Match each RGB image to its mask by filename. PNG masks are single-channel images with background value 0 and crop-row value 255; divide by 255 when binary 0/1 targets are needed.

The scripts download and prepare the dataset independently of any model or training framework. The source archives contain RGB images; the preparation script generates RGB/mask pairs without depth images.

## Tests

Run the preparation tests with:

```bash
python -m unittest discover -s tests -v
```
