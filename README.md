# STEm-Seg
This repository contains the official inference and training implementation for the paper:

**STEm-Seg: Spatio-temporal Embeddings for Instance Segmentation in Videos**

*Ali Athar\*, Sabarinath Mahadevan\*, Aljoša Ošep, Laura Leal-Taixé, Bastian Leibe*

ECCV 2020 | [Paper](https://arxiv.org/abs/2003.08429) | [Video](https://youtu.be/E2Z-1HNO934) | [Project Page](https://www.vision.rwth-aachen.de/publication/00202/)

![TeaserImage](https://github.com/sabarim/STEm-Seg/blob/master/.images/teaser.gif)

## Pre-requisites

* Python 3.7
* PyTorch 1.4, 1.5 or 1.6
* OpenCV, numpy, imgaug, pillow, tqdm, pyyaml, tensorboardX, scipy, pycocotools (see `requirements.txt` for exact versions in case you encounter issues)

## Basic Setup

1. Clone the repository and add append it to the `PYTHONPATH` variable:

   ```bash
   git clone https://github.com/sabarim/STEm-Seg.git
   cd STEm-Seg
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

   
2. Download the required datasets from their respective websites and the trained model checkpoints from the given links. For inference, you only need the validation sets of the target dataset. For training, the table below shows which dataset(s) you will need:

    | Target Dataset        | Datasets Required for Training  | Model Checkpoint |
    |-----------------------| -------------------------------|--------------|
    | [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip)                 | DAVIS'17, YouTubeVIS, COCO Instance Segmentation, PascalVOC | [link](https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/models/davis.pth)
    | [YouTube-VIS](https://competitions.codalab.org/competitions/20128#participate-get_data)           | YouTube-VIS, COCO Instance Segmentation, PascalVOC | [link](https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/models/youtube_vis.pth)
    | [KITTI-MOTS](https://www.vision.rwth-aachen.de/page/mots)            | Mapillary images, KITTI-MOTS, sequence `0002` from [MOTSChallenge](https://motchallenge.net/data/MOTS/) | [link](https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/models/kitti_mots.pth)
    

## Environment Variables

File paths to datasets and model checkpoints are configured using environment variables.


### Required
   
1. `STEMSEG_JSON_ANNOTATIONS_DIR`: To streamline the code, we reorganized the annotations and file paths for every dataset into a standard JSON format. These JSON files can be downloaded from [here](https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/). Set this variable to the directory holding these JSON files.

2. `STEMSEG_MODELS_DIR`: Base directory where models are saved to by default. Only required for training. You can initially point this to any empty directory.

   
   
### Dataset Specific

For inference, you only need to set the relevant variable for the target dataset. For training, since multiple datasets are used, multiple variables will be required (as mentioned below).

#### Video Datasets
   
1. `DAVIS_BASE_DIR`: Set this to the full path of the `JPEGImages/480p` directory for the DAVIS dataset. The image frames for all 60 training and 30 validation videos should be present in the directory. This variable is required for training/inference on DAVIS'19 Unsupervised.
  
2. `YOUTUBE_VIS_BASE_DIR`: Set this to the parent directory of the `train` and `val` directories for the YouTube-VIS dataset. This variable is required for training/inference on YouTube-VIS and also for training for DAVIS.
   
3. `KITTIMOTS_BASE_DIR`: Set this to the `images` directory which contains the directories holding images for each video sequence.


#### Image Datasets (required only for training)

4. `COCO_TRAIN_IMAGES_DIR`: Set this to the `train2017` directory of the COCO instance segmentation dataset. Remember to use the 2017 train/val split and not the 2014 one. This variable is required for training for DAVIS and YouTube-VIS.
   
5. `PASCAL_VOC_IMAGES_DIR`: Set this to the `JPEGImages` directory of the PascalVOC dataset.This variable is required for training for DAVIS and YouTube-VIS. 
   
6. `MAPILLARY_IMAGES_DIR`: You will need to do two extra things here: (1) Put all the training and validation images into a single directory (18k + 2k = 20k images in total). (ii) Since Mapillary images are very large, we first down-sampled them. The expected size for each image is given in `stemseg/data/metainfo/mapillary_image_dims.json` as a dictionary from the image file name to a (width, height) tuple. Please use OpenCV's `cv2.resize` method with `interpolation=cv2.INTER_LINEAR` to ensure the best consistency between your down-sampled images and the annotations we provide in our JSON file. This variable is required for training for KITTI-MOTS.


## Inference

Assuming the relevant dataset environment variables are correctly set, just run the following commands:

1. DAVIS:

    ```bash
    python stemseg/inference/main.py /path/to/downloaded/checkpoints/davis.pth -o /path/to/output_dir --dataset davis
    ```
    
2. YouTube-VIS:

    ```bash
    python stemseg/inference/main.py /path/to/downloaded/checkpoints/youtube_vis.pth -o /path/to/output_dir --dataset ytvis --resize_embeddings
    ```
    
3. KITTI-MOTS:

    ```bash
    python stemseg/inference/main.py /path/to/downloaded/checkpoints/kitti_mots.pth -o /path/to/output_dir --dataset kittimots --max_dim 1948
    ``` 

For each dataset, the output written to `/path/to/output_dir` will be in the same format as that required for the official evaluation tool for each dataset. To obtain visualizations of the generated segmentation masks, you can add a `--save_vis` flag to the above commands.


## Training

1. Make sure the required environment variables are set as mentioned in the above sections. 

2. Run `mkdir $STEMSEG_MODELS_DIR/pretrained` and place the [pre-trained backbone file](https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/models/mask_rcnn_R_101_FPN_backbone.pth) in this directory.

3. **Optional**: To verify if the data loading pipeline is correctly configured, you can separately visualize the training clips by running `python stemseg/data/visualize_data_loading.py` (see `--help` for list of options). 

### DAVIS

The final inference reported in the paper is done using clips of length 16 frames. Training end-to-end with such lengthy clips requires too much GPU VRAM though, so we train in two steps:

1. First we train end-to-end with 8 frame long clips:
   
   ```bash 
   python stemseg/training/main.py --model_dir some_dir_name --cfg davis_1.yaml
   ```
   
2. Then we freeze the encoder network (backbone and FPN) and train only the decoders with 16 frame long clips:

   ```bash 
   python stemseg/training/main.py --model_dir another_dir_name --cfg davis_2.yaml --initial_ckpt /path/to/last/ckpt/from/previous/step.pth
   ```
   
The training code creates a directory at `$STEMSEG_MODELS_DIR/checkpoints/DAVIS/some_dir_name` and places all checkpoints and logs for that training session inside it. For the second step we want to restore the final weights from the first step, hence the additional `--initial_ckpt` argument.

### YouTube-VIS

Here, the final inference was done on 8 frame clips, so the model can be trained in a single step:

```bash 
python stemseg/training/main.py --model_dir some_dir_name --cfg youtube_vis.yaml
```

The training output for this will be placed in `$STEMSEG_MODELS_DIR/checkpoints/youtube_vis/some_dir_name`.

### KITTI-MOTS

Here as well, the final inference was done on 8 frame clips, but we trained in two steps.

1. First on augmented images from the Mapillary dataset:

   ```bash 
   python stemseg/training/main.py --model_dir some_dir_name --cfg kitti_mots_1.yaml
   ```
   
2. Then on the KITTI-MOTS dataset itself:

   ```bash 
   python stemseg/training/main.py --model_dir another_dir_name --cfg kitti_mots_2.yaml --initial_ckpt /path/to/last/ckpt/from/previous/step.pth
   ```
   
   For this step, we included video sequence `0002` from the [MOTSChallenge](https://motchallenge.net/data/MOTS/) training set into our training set. Simply copy the images directory for this video to `$KITTIMOTS_BASE_DIR` and rename the directory to `0050` (this is done because a video named `0002` already exists in KITTI-MOTS). 

### Further Notes on Training

* In general, you will need at least 16GB VRAM for training any of the models. The VRAM requirement can be lowered by reducing the image dimensions in the config YAML file (`INPUT.MIN_DIM` and `INPUT.MAX_DIM`). Alternatively, you can also use mixed precision training by installing [Nvidia apex](https://github.com/NVIDIA/apex) and setting the `TRAINING.MIXED_PRECISION` option in the config YAML to true. In general, both these techniques will reduce performance.

* Multi-GPU training is possible and has been implemented using `torch.nn.parallel.DistributedDataParallel` with one GPU per process. To utilize multiple GPUs, the above commands have to be modified as follows:

  ```bash
  python -m torch.distributed.launch --nproc_per_node=<num_gpus> stemseg/training/main.py --model_dir some_dir_name --cfg <dataset_config.yaml> --allow_multigpu
  ```

* You can visualize the training progress using tensorboard by pointing it to the `logs` sub-directory in the training directory. 

* By default, checkpoints are saved every 10k iterations, but this frequency can be modified using the `--save_interval` argument.

* It is possible to terminate training and resume from a saved checkpoint by using the `--restore_session` argument and pointing it to the full path of the checkpoint.

* We fix all random seeds prior to training, but the results reported in the paper may not be exactly reproducible when you train the model on your own. 

* Run `python stemseg/training/main.py --help` for the full list of options.

## Implementing Other Datasets

Extending the training/inference to other datasets should be easy since most of the code is dataset agnostic.

### Inference

See the if/else block in the `main` method in `inference/main.py`. You will just have to implement a class that converts the segmentation masks produced by the framework to whatever format you want (see any of the scripts in `stemseg/inference/output_utils` for examples).

### Training

You will first have to convert the annotations for your dataset to the standard JSON format used by this code. Inspect any of the given JSON files to see what the format should be like. The segmentation masks are encoded in RLE format using [pycocotools](https://pypi.org/project/pycocotools/). To better understand the file format, you can also see `stemseg/data/generic_video_dataset_parser.py` and `stemseg/data/generic_image_dataset_parser.py` where these files are read and parsed.

Once this is done, you can utilize the `VideoDataset` API in `stemseg/data/video_dataset.py` to do most of the pre-processing and augmentations. You just have to inherit this class and implement the `parse_sample_at` method (see `stemseg/data/davis_data_loader.py` for an example of how to do this).

## Cite

Use the following BibTeX to cite our work:

```
@inproceedings{Athar20ECCV,
  title={STEm-Seg: Spatio-temporal Embeddings for Instance Segmentation in Videos},
  author={Athar, Ali and Mahadevan, Sabarinath and O{\v{s}}ep, Aljo{\v{s}}a and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  booktitle={ECCV},
  year={2020}
}
```
