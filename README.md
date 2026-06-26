# CropRow Detection Lincoln Dataset (CRDLD)

Here we will release the version 2.1 of CRDLD dataset. <br>
[Download dataset](https://lcas.lincoln.ac.uk/nextcloud/index.php/s/Eip4nWbetxJQ6No)
---
[Click here for version 1.0.](https://github.com/JunfengGaolab/CropRowDetection/tree/CRDLDv1) 

---
## Dataset Structure
The dataset is based on 2000 images. The 2000 image dataset is classified into three groups for the purpose of training, testing and validation. The training, testing and validation sub-groups contain 1250,500 and 250 images respectively. Each image in the dataset consists of a corresponding ground truth image. Numeric values of the labelled coordinates are stored in *.mat* files. 

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
The crop row is labelled with white lines on black background. The line width of the white line is 6 pixels. The labels could be regenerated with custom line width using the labels *.mat* files.

<figure>
  <img src="metadata/lbl.png" alt="Alt Text" width="60%">
  <figcaption>Figure 2: Example image and semantic label from the dataset. </figcaption>
</figure>

---

> **Note: Generating Ground Truth Masks**
>
> The labels were created using the MATLAB Video Labeler tool. If you need the **unresized original images** or to regenerate the ground truth image masks:
>
> 1. **Download the unresized original images**:
>    - **[Train Dataset](https://drive.google.com/file/d/1-11NX4hdVav5s6Wf0naG6m6Juup2w4HH/view?usp=sharing)**
>    - **[Test Dataset](https://drive.google.com/file/d/1DyjCnJdRVWxN51DKG2NQKVlbsMl9HLKv/view?usp=sharing)**
>
> 2. Use the provided [MATLAB Code](https://github.com/JunfengGaolab/CropRowDetection/blob/main/codes/label_image_gen.m) to generate the masks.  
>    - Uncomment the **“Fix missing paths if needed”** section to update the directory of the original images on your machine.  
>    - Load the corresponding MATLAB `.mat` data file **before each execution**.  
>    - Provide the correct original image directory **for every run**.
>
> Following these steps is required to correctly generate the ground truth masks.
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
