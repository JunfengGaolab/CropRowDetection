# CRDLD v3.0 — CropRow Detection Lincoln Dataset

CRDLD v3.0 provides sugar beet and maize images with LabelMe JSON annotations and Python tools for generating crop-row masks. <br>
[Download sugar beet and maize originals (Google Drive)](https://drive.google.com/drive/folders/1QrL0-bGUCSY1T2mACdZKVKDcCV8zIeDj?usp=sharing)

Use the [Python preparation scripts](codes/README.md) to download these archives and generate paired RGB images and row masks from the JSON annotations in `labels/`.
---
[Click here for version 1.0.](https://github.com/JunfengGaolab/CropRowDetection/tree/CRDLDv1) 

---
## Historical Dataset Structure
The dataset is based on 2000 images. The 2000 image dataset is classified into three groups for the purpose of training, testing and validation. The training, testing and validation sub-groups contain 1250,500 and 250 images respectively. Each image in the dataset consists of a corresponding ground truth image. The original release stored labelled coordinates in MATLAB files. The sugar beet and maize annotations supplied here are now per-image LabelMe JSON files, compressed in `labels/labels_sb_json.zip` and `labels/labels_maize_json.zip`.

The filenames in the training and testing splits are ordered sequentially by the data class IDs listed in Table 2. In the **Train Dataset**, each class contains 25 images, so images 0–24 correspond to Class 1, images 25–49 to Class 2, and so on. In the **Test Dataset**, each class contains 10 images, so images 0–9 correspond to Class 1, images 10–19 to Class 2, and so on.

    .
    │
    ├── Train Dataset              # 1250 images and ground truth segmentations from 50 data classes. [25 Images per class x 50 Classes]
    |   └──labels.mat             # .mat file containing image label coordinates for 1250 images
    │
    ├── Test Dataset               # 500 images and ground truth segmentations from 50 data classes. [10 Images per class x 50 Classes]
    |    └──Labels                  # .mat file containing image label coordinates for 500 images
    │
    └── Validation Dataset               # 250 images and ground truth segmentations from CRDLDv1.0 base images.
        └──Labels                  # .mat file containing image label coordinates for 250 images

## Table 1: Data Categories
Table 1 contains the different field variations represented in the dataset.

| **ID** | **Data Category** | **Description** |
|--------|-------------------|-----------------|
| a      | Horizontal Shadow | Shadow falls perpendicular to the direction of the crop row |
| b      | Front Shadow      | Shadow of the robot falling on the image captured by the camera |
| c      | Small Crops       | Crop rows at early growth stages (Up to 4 unfolded leaves) |
| d      | Large Crops       | Presence of one or many largely grown crops (more than 4 unfolded leaves) within the crop row |
| e      | Sparse Weed       | Sparsely grown weed scattered between the crop rows |
| f      | Dense Weed        | Weed grown densely among the crop rows where the inter-row space is completely covered |
| g      | Sunny             | Crop row data captured in sunny weather |
| h      | Cloudy            | Crop row data captured in cloudy weather |
| i      | Discontinuities   | Missing plants in the crop row which lead to discontinuities in crop row |
| j      | Slope/ Curve      | Images captured while the crop row is not in a flat farmland or where crop rows are not straight lines |
| k      | Tyre Tracks       | Tyre tracks from tramlines running through the field |

<figure>
  <img src="metadata/cat.png" alt="Alt Text" width="80%">
  <figcaption>Figure 1: Example images from data categories in Table 1. </figcaption>
</figure>

## Table 2: Data Classes Formed by Combinations of Field Variations
Table 2 contains the combinations field variations given in Table 1 in the dataset.

| Data Category      | a | b | c      | d     | e | f | g | h | i | j |
|--------------------|---|---|--------|-------|---|---|---|---|---|---|
| a. Horizontal Shadow | 1 |   |        |       |   |   |   |   |   |   |
| b. Front Shadow      |   | 7 |        |       |   |   |   |   |   |   |
| c. Small Crops       | 2 | 8 | 11, 44 |       |47 |   |   |   |45 |46 |
| d. Large Crops       | 3 | 9 | 12     | 20, 48|50 |   |   |   |49 |   |
| e. Sparse Weed       |   |   | 13     | 21    |   |   |   |   |   |   |
| f. Dense Weed        |   |   | 14     | 22    |   |   |   |   |   |   |
| g. Sunny             |   |   | 15     | 23    |28 |32 |   |   |   |   |
| h. Cloudy            |   |   | 16     | 24    |29 |33 |   |   |   |   |
| i. Discontinuities   | 4 | 10| 17     | 25    |30 |34 |36 |39 |   |   |
| j. Slope/ Curve      | 5 |   | 18     | 26    |31 |35 |37 |40 |42 |   |
| k. Tyre Tracks       | 6 |   | 19     | 27    |   |   |38 |41 |43 |   |


## Sample Data
The crop row is labelled with white lines on black background. The line width of the white line is 6 pixels. Regenerate masks at the original image size, or a chosen height and width, using `python codes/prepare_dataset.py --line-width 6`; see the [preparation guide](codes/README.md).

<figure>
  <img src="metadata/lbl.png" alt="Alt Text" width="60%">
  <figcaption>Figure 2: Example image and semantic label from the dataset. </figcaption>
</figure>

---

## Preparing the sugar beet and maize data

```bash
python -m pip install -r codes/requirements-data.txt
python codes/prepare_dataset.py
# Optional: fixed output size and row width in output pixels
python codes/prepare_dataset.py --height 512 --width 512 --line-width 6 --output /path/to/data
```

The default command downloads both crops from Google Drive, reads images directly from the temporary ZIPs, and reads the matching JSON definitions from `labels/`. It writes `data/<crop>/<train|test>/{image,label}/` at the repository root and deletes downloaded ZIPs afterward. It preserves original image sizes by default. PNG masks contain white rows (255) on a black background (0).

The converted annotations cover **1,505 sugar beet images** (1,075 train, 430 test) and **245 maize images** (175 train, 70 test). They describe the original images in the linked Drive folder; the historical augmented split counts above describe earlier releases.

The first annotated row is the center row. Side-row order varies between images, so JSON shapes retain their original order and 1-based `group_id`, with spatial labels such as `center_row`, `left_row_1`, and `right_row_1`. Side ranks count outward from the center. See [row-order details and loader examples](codes/README.md).

The [MATLAB mask generator](codes/label_image_gen.m) remains available for historical MATLAB labels. Its Python replacement draws polylines directly without MATLAB figures or temporary JPEG screenshots.

---

## Citation

```
@article{de2022deep,
  title={Deep learning-based Crop Row Following for Infield Navigation of Agri-Robots},
  author={de Silva, Rajitha and Cielniak, Grzegorz and Wang, Gang and Gao, Junfeng},
  journal={Journal of Field Robotics},
  year={2023}
}
```
