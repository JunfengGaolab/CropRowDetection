# CropRow Detection with Unet

Here we will release the dataset for the submission ([Towards agricultural autonomy: crop row detection under varying field conditions using deep learning](http://arxiv.org/abs/2109.08247)) in the ICRA 2022.

---
## Dataset Structure
The dataset is based on 500 base images. The 500 base images are augmented into 2000 sub-images. The 2000 image dataset is classified into two groups for the purpose of testing and training. Each training and testing sub-groups contains 1000 images. Each image in the dataset consists of a corresponding ground truth image. 

Numeric values of the labelled coordinates are stored in *.mat* files. All the label coordinates in train dataset are stored in a single *.mat* file. The label coordinates in test dataset are stored in 10 separate *.mat* files.

    .
    │
    ├── Train Dataset              # 1000 images based on 250 base images. [100 Images per category x 10 Categories]
    |   ├── labels.mat             # .mat file containing image label coordinates for 250 base images
    │
    └── Test Dataset               # 1000 images based on 250 base images
        ├── 250 Resized images     # 250 base images are resized into 512x512 resolution
        ├── 10 Data Categories     # 1000 images with 512x512 resolution
        │  ├── Horizontal Shadows  # Shadow  falls  perpendicular  to  the  direction of the crop row
        │  ├── Slope/ Curve        # Images captured while the crop row is not in a flat farmland or where crop rows are not straight lines
        │  ├── Discontinuities     # Missing  plants  in  the  crop  row  which leads to discontinuities in crop row
        │  ├── FrontShadow         # Shadow of the robot falling on the image captured by the camera
        │  ├── Dense Weed          # Weed grown densely among the crop rows
        │  ├── Large Crops         # Presence  of  one  or  many  largely  grown crops within the crop row
        │  ├── Small Crops         # Crop rows at early growth stages
        │  ├── Sunlight            # Sunlight  falling  on  the  camera  causing lens flares and similar distortions
        │  ├── Tyre Tracks         # Tyre    tracks    from    tramlines    running through the field
        │  └── Sparse Weed         # Sparsely  grown  weed  scattered  between the crop rows
        └──Labels                  # 10 .mat files containing image label coordinates for 25 base images per category

## Data Augmentation
Base image is augmented into four sub images by cropping and rotating in different orientations.

![metadata/cropping.jpg](metadata/cropping.jpg)

## Sample Data
The crop row is labelled with white lines on black background. The line width of the white line is 8 pixels. The labels could be regenerated with custom line width using the labels *.mat* files.

![metadata/DataSample.jpg](metadata/DataSample.jpg)


## Citation

```
@misc{desilva2021agricultural,
      title={Towards agricultural autonomy: crop row detection under varying field conditions using deep learning},
      author={Rajitha de Silva and Grzegorz Cielniak and Junfeng Gao},
      year={2021},
      eprint={2109.08247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
