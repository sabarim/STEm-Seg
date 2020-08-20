from stemseg.training.exponential_lr import ExponentialLR
from stemseg.config import cfg
from stemseg.data import CocoDataLoader, YoutubeVISDataLoader, DavisDataLoader, MapillaryDataLoader, MOTSDataLoader, \
    PascalVOCDataLoader
from stemseg.data import CocoPaths, YoutubeVISPaths, MapillaryPaths, DavisUnsupervisedPaths as DavisPaths, \
    KITTIMOTSPaths, PascalVOCPaths
from stemseg.data.concat_dataset import ConcatDataset as CustomConcatDataset
from stemseg.utils import LossConsts

from stemseg.data.distributed_data_sampler import DistributedSampler as CustomDistributedSampler
from stemseg.data.iteration_based_batch_sampler import IterationBasedBatchSampler
from stemseg.utils import distributed as dist_utils

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler

import torch
import torch.optim.lr_scheduler as lrs
import logging


_VAR_KEY_TO_DISP_STR = {
    LossConsts.EMBEDDING: "EmbL",
    LossConsts.SEMSEG: "SegL",
    LossConsts.VARIANCE_SMOOTHNESS: "VarS",
    LossConsts.LOVASZ_LOSS: "LovL",
    LossConsts.SEEDINESS_LOSS: "SeedL",
    LossConsts.FOREGROUND: "FgL",
}


def var_keys_to_str(losses):
    s = ''
    for k, v in losses.items():
        if k == "lr":
            s += "LR: {:.2E} - ".format(v)
        else:
            s += "{:s}: {:.3f} - ".format(_VAR_KEY_TO_DISP_STR[k], v)
    return s[:-3]


def register_log_level_type(parser):
    def str2LogLevel(v):
        if v.lower() == "fatal":
            return logging.FATAL
        elif v.lower() == "critical":
            return logging.CRITICAL
        elif v.lower() == "error":
            return logging.ERROR
        elif v.lower() in ("warn", "warning"):
            return logging.WARN
        elif v.lower() in ("info", "normal"):
            return logging.INFO
        elif v.lower() in ("debug", "verbose"):
            return logging.DEBUG
        elif v.lower() in ("no", "false", "off", "f", "0"):
            return False
        else:
            raise ValueError("Failed to cast '{}' to logging level".format(v))

    parser.register('type', 'LogLevel', str2LogLevel)
    return "LogLevel"


def create_concat_dataset_for_davis(total_samples, print_fn):
    if print_fn is None:
        print_fn = print

    print_fn("Creating training dataset for Davis...")
    assert cfg.INPUT.NUM_CLASSES == 2
    datasets = []
    ds_weights = []
    ds_names = []

    ds_cfg = cfg.DATA.DAVIS

    # Coco
    datasets.append(CocoDataLoader(CocoPaths.images_dir(), CocoPaths.ids_file(), category_agnostic=True))
    ds_weights.append(ds_cfg.COCO_WEIGHT)
    ds_names.append("Coco")

    # YoutubeVIS
    num_subseqs = int(round(total_samples * ds_cfg.YOUTUBE_VIS_WEIGHT))
    datasets.append(YoutubeVISDataLoader(YoutubeVISPaths.training_base_dir(), YoutubeVISPaths.train_vds_file(),
                                         num_subseqs, category_agnostic=True,
                                         single_instance_duplication=cfg.DATA.YOUTUBE_VIS.SINGLE_INSTANCE_DUPLICATION))
    ds_weights.append(ds_cfg.YOUTUBE_VIS_WEIGHT)
    ds_names.append("YouTubeVIS")

    # Davis
    num_subseqs = int(round(
        cfg.TRAINING.MAX_ITERATIONS * cfg.TRAINING.BATCH_SIZE * ds_cfg.DAVIS_WEIGHT))
    datasets.append(DavisDataLoader(
        DavisPaths.trainval_base_dir(), DavisPaths.train_vds_file(),
        samples_to_create=num_subseqs,
        single_instance_duplication=True,
        background_as_ignore_region=True
    ))
    ds_weights.append(ds_cfg.DAVIS_WEIGHT)
    ds_names.append("Davis")

    # PascalVOC
    datasets.append(PascalVOCDataLoader(PascalVOCPaths.images_dir(), PascalVOCPaths.ids_file(),
                                        category_agnostic=True))
    ds_weights.append(ds_cfg.PASCAL_VOC_WEIGHT)
    ds_names.append("PascalVOC")

    print_fn("Training datasets: {}".format(', '.join(ds_names)))

    return CustomConcatDataset(datasets, total_samples, ds_weights)


def create_concat_dataset_for_youtube_vis(total_samples, print_fn):
    if print_fn is None:
        print_fn = print

    print_fn("Creating training dataset for YouTube-VIS...")

    assert cfg.INPUT.NUM_CLASSES == 41  # 40 classes + 1 generic foreground class
    datasets = []
    ds_weights = []
    ds_names = []

    ds_cfg = cfg.DATA.YOUTUBE_VIS

    # Coco
    datasets.append(CocoDataLoader(CocoPaths.images_dir(), CocoPaths.ids_file(), category_agnostic=False))
    ds_weights.append(ds_cfg.COCO_WEIGHT)
    ds_names.append("Coco")

    # PascalVOC
    datasets.append(PascalVOCDataLoader(PascalVOCPaths.images_dir(), PascalVOCPaths.ids_file(),
                                        category_agnostic=False))
    ds_weights.append(ds_cfg.PASCAL_VOC_WEIGHT)
    ds_names.append("PascalVOC")

    # YoutubeVIS
    num_subseqs = int(round(total_samples * ds_cfg.YOUTUBE_VIS_WEIGHT))
    datasets.append(
        YoutubeVISDataLoader(YoutubeVISPaths.training_base_dir(), YoutubeVISPaths.train_vds_file(),
                             num_subseqs,
                             category_agnostic=False,
                             single_instance_duplication=cfg.DATA.YOUTUBE_VIS.SINGLE_INSTANCE_DUPLICATION))
    ds_weights.append(ds_cfg.YOUTUBE_VIS_WEIGHT)
    ds_names.append("YouTubeVIS")

    print_fn("Training datasets: {}".format(', '.join(ds_names)))
    return CustomConcatDataset(datasets, total_samples, ds_weights)


def create_concat_dataset_for_kitti_mots(total_samples, print_fn=None):
    if print_fn is None:
        print_fn = print

    print_fn("Creating training dataset for KITTI-MOTS...")
    assert cfg.INPUT.NUM_CLASSES == 3  # car, pedestrian, background
    datasets = []
    ds_weights = []
    ds_names = []

    ds_cfg = cfg.DATA.KITTI_MOTS

    # Mapillary
    if ds_cfg.MAPILLARY_WEIGHT > 0.:
        datasets.append(MapillaryDataLoader(MapillaryPaths.images_dir(), MapillaryPaths.ids_file()))
        ds_weights.append(ds_cfg.MAPILLARY_WEIGHT)
        ds_names.append("Mapillary")

    # KITTI-MOTS
    if ds_cfg.KITTI_MOTS_WEIGHT > 0.:
        num_subseqs = int(round(total_samples * ds_cfg.KITTI_MOTS_WEIGHT))
        datasets.append(MOTSDataLoader(
            KITTIMOTSPaths.train_images_dir(), KITTIMOTSPaths.train_vds_file(), num_subseqs))
        ds_weights.append(ds_cfg.KITTI_MOTS_WEIGHT)
        ds_names.append("KITTI-MOTS")

    print_fn("Training datasets: {}".format(', '.join(ds_names)))

    return CustomConcatDataset(datasets, total_samples, ds_weights)


def create_training_dataset(total_samples, print_fn=None):
    dataset_creation_fns = {
        "davis": create_concat_dataset_for_davis,
        "youtube_vis": create_concat_dataset_for_youtube_vis,
        "kitti_mots": create_concat_dataset_for_kitti_mots,
    }

    try:
        return dataset_creation_fns[cfg.TRAINING.MODE](total_samples, print_fn)
    except KeyError as _:
        raise ValueError("Invalid training mode: {}".format(cfg.TRAINING.MODE))


def create_optimizer(model, cfg, print_fn=None):
    if print_fn is None:
        print_fn = print

    if cfg.OPTIMIZER.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), cfg.INITIAL_LR, cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY, nesterov=cfg.NESTEROV)
        print_fn("Using SGD optimizer with momentum {} and weight decay {}".format(cfg.MOMENTUM, cfg.WEIGHT_DECAY))
    elif cfg.OPTIMIZER.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), cfg.INITIAL_LR, weight_decay=cfg.WEIGHT_DECAY)
        print_fn("Using Adam optimizer with weight decay {}".format(cfg.WEIGHT_DECAY))
    else:
        raise ValueError("Invalid optimizer choice: '{}'".format(cfg.OPTIMIZER))

    return optimizer


def create_lr_scheduler(optimizer, cfg, print_fn=None):
    if print_fn is None:
        print_fn = print

    if cfg.LR_DECAY_TYPE == "step":
        lr_scheduler = lrs.MultiStepLR(optimizer, cfg.LR_DECAY_STEPS, cfg.LR_DECAY_FACTOR)
        print_fn("Multistep LR decay at {} steps with decay factor {}".format(cfg.LR_DECAY_STEPS, cfg.LR_DECAY_FACTOR))
    elif cfg.LR_DECAY_TYPE == "exponential":
        lr_scheduler = ExponentialLR(optimizer, cfg.LR_EXP_DECAY_FACTOR, cfg.LR_EXP_DECAY_STEPS, cfg.LR_EXP_DECAY_START)
        print_fn("Exponential decay starting at {} steps, lasting {} steps, with decay factor {}".format(
            cfg.LR_EXP_DECAY_START, cfg.LR_EXP_DECAY_STEPS, cfg.LR_EXP_DECAY_FACTOR))
    elif cfg.LR_DECAY_TYPE == "none":
        lr_scheduler = lrs.LambdaLR(optimizer, lambda step: 1.0)
        print_fn("Learning rate decay is disabled.")
    else:
        raise ValueError("Invalid learning rate decay type: {}".format(cfg.LR_DECAY_TYPE))

    print_fn("{} optimizer created with initial learning rate {}.".format(cfg.OPTIMIZER, cfg.INITIAL_LR))

    return lr_scheduler


def create_training_data_loader(dataset, batch_size, shuffle, collate_fn=None, num_workers=0, elapsed_iters=0):
    is_distributed = dist_utils.is_distributed()
    if is_distributed:
        sampler = CustomDistributedSampler(dataset, dist_utils.get_world_size(), dist_utils.get_rank(), shuffle)
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    if elapsed_iters > 0:
        print("Elapsed iters: {}".format(elapsed_iters))
        batch_sampler = IterationBasedBatchSampler(batch_sampler, int(len(dataset) / batch_size), elapsed_iters)

    return DataLoader(dataset,
                      collate_fn=collate_fn,
                      batch_sampler=batch_sampler,
                      num_workers=num_workers)
