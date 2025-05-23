# Aerial-MegaDepth Data Generation

We provide data and pipeline for generating our AerialMegaDepth dataset using Google Earth and MegaDepth. It includes a minimal example as well as instructions for generating your own data from scratch.

## Table of Contents
- [📦 Sample Data](#-sample-data)  
  - [Download via CLI](#download-via-cli)  
  - [Sample Data Structure](#sample-data-structure)  
- [🛠️ Generating Data from Scratch](#️-generating-data-from-scratch)  
  - [0️⃣ Prerequisites](#0️⃣-prerequisites)  
  - [1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio](#1️⃣-generating-pseudo-synthetic-data-from-google-earth-studio)  
  - [2️⃣ Registering to MegaDepth](#2️⃣-registering-to-megadepth)   
  - [🧪 (Optional) Prepare Data for Training DUSt3R/MASt3R](#🧪-optional-prepare-data-for-training-dust3rmast3r)  
- [License](#license)


## 💡 Before you start...
> For the following commands, `/mnt/slarge2/` is our local directory path where we store the data. You should replace it with the appropriate path on your machine.

> If you run into issues preparing the dataset or are working on a research project that could benefit from our training data (particularly for academic use), feel free to reach out to me via [email](mailto:kvuong@andrew.cmu.edu). I'll do my best to help!

## 📦 Sample Data

We provide a sample scene (`0001`) to illustrate the format and structure of the dataset. You can download it directly using the AWS CLI.

### Download via CLI

You can use [AWS CLI](https://aws.amazon.com/cli/) to download the sample data:

```bash
mkdir -p /mnt/slarge2/megadepth_aerial_data/data
aws s3 sync s3://aerial-megadepth/full_data/0001 /mnt/slarge2/megadepth_aerial_data/data/0001
```
This command will download the sample scene data to `/mnt/slarge2/megadepth_aerial_data/data/0001`.

### Sample Data Structure

```
megadepth_aerial_data/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                └── localized_dense_metric/
                    ├── images/           # RGB images (Google Earth & MegaDepth)
                    ├── depths/           # Depth maps
                    └── sparse-txt/       # COLMAP reconstruction files
```

## 🛠️ Generating Data from Scratch

The full pipeline involves two stages:

1. [Generating Pseudo-Synthetic Data](#1-generating-pseudo-synthetic-data-from-google-earth-studio)  
2. [Registering to MegaDepth](#2-registering-to-megadepth)

### 0️⃣ Prerequisites
We provided a `.npz` file containing a list of scenes and images from MegaDepth in `datasets_preprocess/megadepth_image_list.npz`. These images will be registered to the corresponding pseudo-synthetic data.

### 1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio

This stage creates video frames and camera metadata using Google Earth Studio.

#### Step 1: Render Using Google Earth Studio

Each scene comes with pre-defined camera parameters in `.esp` format. You can download all `.esp` files using:

```bash
aws s3 sync s3://aerial-megadepth/geojsons /mnt/slarge2/megadepth_aerial_data/geojsons
```

Directory structure:

```
megadepth_aerial_data/
└── geojsons/
    ├── 0001/
    │   └── 0001.esp
    ├── 0002/
    │   └── 0002.esp
    └── ...
```

To render the pseudo-synthetic sequence:

1. Open [Google Earth Studio](https://earth.google.com/studio/)
2. Import a `.esp` file via **File → Import → Earth Studio Project**
3. Go to **Render** and export:
   - **Video**: select **Cloud Rendering** to produce a `.mp4`
   - **Tracking Data**: enable **3D Camera Tracking (JSON)** with **Coordinate Space: Global**

Save the exported files to:

```
megadepth_aerial_data/
└── downloaded_data/
    ├── 0001.mp4     # Rendered video
    ├── 0001.json    # Camera metadata (pose, intrinsics, timestamps)
    └── ...
```

> 💡 Note: This step currently requires manual interaction with Google Earth Studio, which can be inconvenient. We actively welcome PRs or discussions that explore ways to automate or streamline this step!

#### Step 2: Extract Images & Metadata

Use the provided script to extract frames from each `.mp4` video and also extract camera metadata from the corresponding `.json` file:

```bash
python datasets_preprocess/preprocess_ge.py \
    --data_root /mnt/slarge2/megadepth_aerial_data \
    --scene_list ./datasets_preprocess/megadepth_image_list.npz
```

This will generate per-scene folders with extracted frames and frame-aligned metadata:

```
megadepth_aerial_data/
└── data/
    ├── 0001/
    │   ├── 0001.json               # Aligned metadata (pose, intrinsics, timestamps)
    │   └── footage/
    │       ├── 0001_000.jpeg       # Extracted video frames
    │       ├── 0001_001.jpeg
    │       └── ...
    └── ...
```


### 2️⃣ Registering to MegaDepth

Once pseudo-synthetic images are generated, the next step is to localize them within a MegaDepth scene and reconstruct the scene geometry.

#### Step 1: Prepare MegaDepth Images

First, download the [MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) by following their instructions. After downloading, your dataset root (e.g., `/mnt/slarge/megadepth_original`) should contain the folders `MegaDepth_v1_SfM` and `phoenix`.

Then, use the provided preprocessing script to extract RGB images, depth maps, and camera parameters for each scene:

```bash
python datasets_preprocess/preprocess_megadepth.py \
    --megadepth_dir /mnt/slarge/megadepth_original/MegaDepth_v1_SfM \
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list.npz \
    --output_dir /mnt/slarge2/megadepth_processed
```

This will generate processed outputs with the following structure:

```text
megadepth_processed/
├── 0001/
│   └── 0/
│       ├── 5008984_74a994ce1c_o.jpg.jpg    # RGB image
│       ├── 5008984_74a994ce1c_o.jpg.exr    # Depth map (EXR format)
│       ├── 5008984_74a994ce1c_o.jpg.npz    # Camera pose + intrinsics
│       └── ...
├── 0002/
│   └── 0/
│       └── ...
└── ...
```

Each `.jpg` file corresponds to a view and is paired with:
- a `.npz` file containing camera intrinsics and extrinsics
- a `.exr` file containing a depth map in metric scale


#### Step 2: Run the Data Registration Pipeline
※ Dependencies can be installed following the instructions in the [hloc repository](https://github.com/cvg/Hierarchical-Localization).


With both pseudo-synthetic frames and preprocessed MegaDepth data prepared, run the localization and reconstruction pipeline using:

```bash
python do_colmap_localization.py \
    --root_dir /mnt/slarge2/megadepth_aerial_data/data \
    --megadepth_dir /mnt/slarge2/megadepth_processed/ \
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list.npz
```

The output is saved per scene as:

```
megadepth_aerial_data/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                └── localized_dense_metric/
                    ├── images/           # Registered RGB images
                    ├── depths/           # MVS depth maps
                    └── sparse-txt/       # COLMAP poses + intrinsics (text format)
```

### 🧪 (Optional) Prepare Data for Training DUSt3R/MASt3R
We provide the precomputed pairs for training DUSt3R/MASt3R. First, download the precomputed pairs:

```bash
mkdir -p data_splits
wget https://aerial-megadepth.s3.us-east-2.amazonaws.com/data_splits/aerial_megadepth_train_part1.npz -P data_splits
wget https://aerial-megadepth.s3.us-east-2.amazonaws.com/data_splits/aerial_megadepth_train_part2.npz -P data_splits
wget https://aerial-megadepth.s3.us-east-2.amazonaws.com/data_splits/aerial_megadepth_val.npz -P data_splits
```

Use the following script to preprocess the data to be compatible with DUSt3R or MASt3R training:

```bash
python datasets_preprocess/preprocess_aerialmegadepth.py \
    --megadepth_aerial_dir /mnt/slarge2/megadepth_aerial_data/data \
    --precomputed_pairs ./data_splits/aerial_megadepth_train_part1.npz \
    --output_dir /mnt/slarge2/megadepth_aerial_processed
```

## License
Google Earth data belong to [Google](https://www.google.com/earth/studio/faq/) and is available for non-commercial research purposes only. For full information, please refer to their [TOS](https://earthengine.google.com/terms/).