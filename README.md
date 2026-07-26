<p align="center">
<img src="./docs/resources/precision_track.png" alt="PrecisionTrack" width="200">
<span style="font-size: 32px; margin: 0 20px; vertical-align: top;">+</span>
<img src="https://github.com/WildlifeDatasets/wildlife-tools/raw/main/docs/resources/tools-logo.png" alt="Wildlife tools" width="150">
</p>

<div align="center">
  <p align="center"><span style="font-size: 16;">A Python toolkit for training custom animal re-identification models. Seamlessly integrate with PrecisionTrack to enable appearance-based identity tracking for any species in your behavioral studies.</span></p>

<p align="center"><span style="font-size: 16;"><a href="https://wildlifedatasets.github.io/wildlife-tools/">Documentation</a></span></p>

</div>

## Introduction

**PrecisionTrack ReID** Enables researchers to train species-specific re-identification models and deploy them. These models can then be used within [PrecisionTrack](https://github.com/VincentCoulombe/precision_track/tree/main), enabling PrecisionTrackers to perform evidence-based re-identification. To configure your PrecisionTracker, you will first need to go over all the steps within the **How to use** section of this README in order to train, test and deploy a re-identification model, then follow the [PrecisionTrack appearance re-identification configuration guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation) to configuration your PrecisionTracker so it uses your newly deployed re-identification model.

## Installation

### CPU installation

1. Create a python virtual environment

```script
conda create -n precision_track_reid python==3.11
```

2. Activate your python virtual environment

```script
conda activate precision_track_reid
```

3. Install the CPU build of PyTorch

```script
pip install torch==2.6.0
```

4. Clone the repository using `git` and install it.

```script
git clone https://github.com/VincentCoulombe/precision_track-ReID.git

cd precision_track-ReID
pip install -e ".[cpu]"
```

### GPU installation

1. Create a python virtual environment

```script
conda create -n precision_track_reid python==3.11
```

2. Activate your python virtual environment

```script
conda activate precision_track_reid
```

3. Install a GPU-enabled build of PyTorch matching your CUDA driver

```script
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**NOTE**: `torch==2.6.0` is only published for the `cu118`, `cu124`, and `cu126` tags.

4. Clone the repository using `git` and install it.

```script
git clone https://github.com/VincentCoulombe/precision_track-ReID.git

cd precision_track-ReID
pip install -e ".[cuda]"
```

## How to use

### 1 Create a MOT dataset

First, you will need a dataset to train, validate and test your re-identification model.
Fortunately, if you already have a PrecisionTracker trained and configured, you will be able to create one almost automatically.
To do so, you need to execute the following steps:

#### 1.1 Creating a dataset root directory:

Create a new folder (it will be referenced to as your dataset root directory from now on). Inside your dataset root directory,
Create the following subdirectories.

```bash
<Your dataset root directory>/
  ├── bboxes/
  │ ├── train/
  │ ├── val/
  ├── videos/
  │ ├── train/
  │ ├── val/
```

**NOTE** You will need to register the path to your dataset root directory inside your `./configs/user_configs.yaml` file. You can refer to our [Config guide](https://github.com/VincentCoulombe/precision_track-ReID/tree/main/configs) for more details.

#### 1.2 Adding your videos inside your dataset root directories:

Move your videos inside your dataset root directories (I added a small example of what it might look like afterward).

**NOTE 1**: Make sure all your videos have different names (will be usefull for a following step).
**NOTE 2**: Make sure your validation videos are different enough from your training videos so the validation process is usefull. For example,
a good practice would be to train and validate on videos coming from recordings that occured on different days.

```bash
<Your dataset root directory>/
  ├── bboxes/
  │ ├── train/
  │ ├── val/
  ├── videos/
  │ ├── train/
  │   ├── video1.mp4
  │   ├── video2.avi
  │   ├── etc...
  │ ├── val/
  │   ├── video3.mp4
  │   ├── etc...
```

**Awesome engineering tip 1**: If your training and validation videos contain marked animals, make sure all your animal are marked. In other words, do not have a non-marked animal as an identity within your dataset. This is because, the evidence-based re-identification tracking system might be confuse between an animal with a non-visible mark and a non-marked animal.

**Awesome engineering tip 2**: While following engineering tip #1 is good practice, we found that training a re-identification system using a non-mark identity, then moving that non-marked identity within the `disabled_identities` list of the [PrecisionTrack appearance re-identification configuration file](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation) yielded the best results. In other words, we purposely added a non-marked mouse into our re-identification dataset's recordings so we could train the re-identification model to learn how to classify non-visible marks. We then configure our PrecisionTracker to ignore those classifications during tracking. Obviously, zero non-marked animals ever entered our vivarium during our actual tracking sessions.

#### 1.3 Creating your bounding boxes .csv files

Now that your dataset have videos of your subjects, you have 1/2 of the information this repository need to train a re-identification model.
The other half is to know "where each subjects are within the videos". To do so, we need to create [MOT-styled annotations](https://motchallenge.net/).
A MOT-styled annotation file is a `.csv` where each row describes a single bounding box in a single frame. For example:

| frame_id | class_id | instance_id | x   | y   | w   | h   | score |
| -------- | -------- | ----------- | --- | --- | --- | --- | ----- |
| 0        | 0        | 2           | 412 | 158 | 64  | 72  | 1     |
| 0        | 0        | 5           | 730 | 220 | 58  | 70  | 1     |
| 1        | 0        | 2           | 415 | 160 | 64  | 71  | 1     |

**NOTE** In the context of the MOT-styled annotations, the column `class_id` refers to the subjects species.

This is where PrecisionTrack come in handy. You cam train a PrecisionTracker to automatically label your videos (thus almost automatically creating your MOT dataset). More specifically, this is how **we created our MOT dataset**:

We created our MOT dataset by doing the following:

1. Train a good (80%+ detection F1 & 80%+ OKS) subject detection and pose-estimation model (follow [PrecisionTrack's documentation](https://github.com/VincentCoulombe/precision_track) for more details)
2. Use the `batch_track_directory.py` tool,
   Please refer to the [tooling documentation](https://github.com/VincentCoulombe/precision_track/tree/main/tools) and the [configuration documentation](https://github.com/VincentCoulombe/precision_track/tree/main/configs). This tool will generate tracking results for every videos in the provided directory. The relevant tracking results (for creating MOT dataset) will be saved in the `<saving_directory>/*/tracked_bboxes.csv` files.
3. Use the `visualize.py` tool (with only `display_bounding_boxes` set to `true` as Visualization parameter) to create a visual for every tracking results obtained in 2).
4. Manually review the tracking results and correct the tracking errors (ID switches).
   You can manually edit the tracking switches by editing the `instance_id` column (meaning switching back the swapped IDs) of the `tracked_bboxes.csv` file saved inside the `saving_directory` from which the visual was created.
5. Re-runn the step 3) to generate up-to-date visuals. These new visuals will take into account your manual corrections. Now you will be able to assess it for a final revision.
6. Move your `<saving_directory>/*/tracked_bboxes.csv` files (which are your MOT annotation files) into their respective places inside `<Your dataset root directory>` and rename them so their names match their correcponding videos.

**Awesome engineering tip 1**: If you have multiple subjects in your videos, make sure to manually correct the potential identity switches PrecisionTrack might have made. To do so, execute the following steps: 1) Use PrecisionTrack's visualize.py tool to have a visual representation of your tracking results. 2) Watch the actual visualization video to check if tracking errors occured. 3) If tracking errors occured, correct them by opening the video's `tracked_bboxes.csv` file in excel and manually correcting the errors (swapping back the instance_id). Repeat step 1 to visualize your manual modifications and repeat if necessary.

**Awesome engineering tip 2**: You can avoid all the trouble described in the top engineering tip by having just one subject per re-identification training video. Yes, the re-identification will still be good in a multi-subject setting even if it was trained on videos containing single animals.

For reference, you can check the [MICE sequential dataset](https://drive.google.com/drive/folders/1WcDkX-92X6SCgZPAZXFyDc6EGUzU0Onq?usp=drive_link) which have MOT-styled annonotations under its `./bboxes/*` directories.

### 1.4 Creating your dataset metadata file

You will also need a dataset metadata file (identities between each of yours MOT bounding box files to your defined class identifiants). This file will be a `.json`. Here's an example of a valid file:

```json
{
	"classes": ["Benjamin", "Jacob", "Noodle", "Puddy", "Leo"],
	// NOTE the sequence mappings are expected to be the same length as the classes list
	"sequence_2025-12-22T09_08_33": [
		[0, 4], // NOTE this means that the identity 'Benjamin' is linked to the instance id 4 of the class id 0
		[0, 1],
		[0, 3],
		[0, 5], // NOTE this means that the identity linked to instance id 5 of the class id 0 is 'Puddy'
		[0, 2]
	],
	"sequence_2026-01-25T10_25_01": [
		[0, 2], // NOTE in this sequence, 'Benjamin' is linked to instance id 2
		[0, -1], // NOTE set to -1 if absent from the .csv file
		[0, 6],
		[0, 5],
		[0, 4]
	]
}
```

**Note** In the context of our re-identification training, the column `classes` refers to the subjects identities.

**Structure:**

- **`classes`**: A list of your dataset's identity names (e.g., individual animal names or IDs).
- **Video entries** (e.g., `sequence_2025-12-22T09_08_33`, `sequence_2026-01-25T10_25_01`): Each key corresponds to a video name and contains mappings between the MOT bounding box file columns:
  - The first value in each pair corresponds to the **label column** (2nd column in the MOT `bboxes.csv` files)
  - The second value corresponds to the **instance ID column** (3rd column in the MOT `bboxes.csv` files)

Each of the **Video entries** mappings are expected to be lists with the same length as the number of classes (identities). Simply mark missing instance ids as `-1`.

This mapping allows the network to associate each unique IDS (a combination of the label and the instance ID of each subjects) of your MOT bounding box files with your defined identities in the `classes` list.

**NOTE** You will need to register the path to your dataset metadata file inside your `./configs/user_configs.yaml` file. You can refer to our [Config guide](https://github.com/VincentCoulombe/precision_track-ReID/tree/main/configs) for more details.

### 2. Train, test and deploy your re-identification model

Congratulation! You can now launch the training, testing and deployment processes with the following commands:

```script
cd ./tools
python train_test_deploy.py
```

This will train, test and deploy your first re-identification model. If you like your test metric values, you can go ahead and track using it, please refer to our [PrecisionTrack appearance re-identification configuration guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation) for more details. If you are not satisfied, you will need to perform one of the following:

- Make sure all your animals are visually distinguishable (if you cannot distinguish them, the model wont). this means that your marked animals need to have their marks being visible at (almost) all time.
- Make sure you have a nice data distribution within your dataset (ensure that your training videos contain visible, moving and active subjects).
- Make sure your annotations are correct (by manually checking the generated `<dataset root directory>/crops/` directory).

### 3. BONUS: Move your deployed checkpoints to your PrecisionTrack deployment directory

It is good practice to move the newly generated `.onnx` (and/or `.engine`) file(s) along with the `.yaml` file your PrecisionTrack's [deploying_directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs).

## Citation

```latex
@misc{precision_track2025,
    title={PrecisionTrack: A Platform for Automated Long-Term Social Behavior Analysis in Naturalized Environments},
    author={Coulombe & al},
    year={2025}
}
```

```
@InProceedings{Cermak_2024_WACV,
    author    = {\v{C}erm\'ak, Vojt\v{e}ch and Picek, Luk\'a\v{s} and Adam, Luk\'a\v{s} and Papafitsoros, Kostas},
    title     = {{WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification}},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5953-5963}
}
```
