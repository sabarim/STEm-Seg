from argparse import ArgumentParser
from stemseg.data import CocoDataLoader, YoutubeVISDataLoader, DavisDataLoader, MOTSDataLoader, MapillaryDataLoader, \
    PascalVOCDataLoader
from stemseg.data.paths import CocoPaths, YoutubeVISPaths, DavisUnsupervisedPaths, KITTIMOTSPaths, \
    MapillaryPaths, PascalVOCPaths
from stemseg.data.common import collate_fn, visualize_semseg_masks, instance_masks_to_semseg_mask
from stemseg.config import cfg
from stemseg.utils import RepoPaths
from stemseg.utils.vis import overlay_mask_on_image, create_color_map

from torch.utils.data import DataLoader as TorchDataLoader

import cv2
import imgaug
import numpy as np
import os
import random
import torch


def visualize_data_loader_output(dataset, num_workers, batch_size, shuffle):
    print("Number of samples: {}".format(len(dataset)))
    data_loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate_fn)

    for t in range(cfg.INPUT.NUM_FRAMES):
        cv2.namedWindow('Image {}'.format(t), cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ignore {}'.format(t), cv2.WINDOW_NORMAL)
        cv2.namedWindow('Image {} semseg'.format(t), cv2.WINDOW_NORMAL)

    cmap = create_color_map().tolist()
    for image_list, target, meta_info in data_loader:
        images = image_list.numpy()

        for i in range(batch_size):
            if meta_info[i]:
                if 'seq_name' in meta_info[i]:
                    print("Seq name: {}".format(meta_info[i]['seq_name']))
                if 'category_labels' in meta_info[i]:
                    print("Category labels: {}".format(str(meta_info[i]['category_labels'])))

            seq_images = [np.copy(images[i][t]) for t in range(cfg.INPUT.NUM_FRAMES)]

            masks = target[i]['masks']
            num_instances = masks.shape[0]
            print("Instances: {}".format(num_instances))

            semseg_mask = instance_masks_to_semseg_mask(masks, target[i]['category_ids'])
            semseg_mask = semseg_mask.numpy().astype(np.uint8)

            ignore_mask = target[i]['ignore_masks']
            seq_images_instances = [im.copy() for im in seq_images]

            for j in range(num_instances):
                masks_j = [m.numpy() for m in masks[j].unbind(0)]

                for t in range(cfg.INPUT.NUM_FRAMES):
                    seq_images_instances[t] = overlay_mask_on_image(seq_images_instances[t], masks_j[t], 0.5, cmap[j + 1])

            seq_image_ignore_overlayed = []
            for t in range(cfg.INPUT.NUM_FRAMES):
                semseg_masked_t = visualize_semseg_masks(images[i][t], semseg_mask[t])
                cv2.imshow('Image {} semseg'.format(t), semseg_masked_t)

                seq_image_ignore_overlayed.append(overlay_mask_on_image(seq_images[t], ignore_mask[t].numpy()))

                cv2.imshow('Image {}'.format(t), seq_images_instances[t])
                cv2.imshow('Ignore {}'.format(t), seq_image_ignore_overlayed[t])

            if cv2.waitKey(0) == 113:  # 'q' key
                return


def main(args):
    imgaug.seed(42)
    torch.random.manual_seed(42)
    random.seed(42)

    if os.path.isabs(args.cfg):
        cfg.merge_from_file(args.cfg)
    else:
        cfg.merge_from_file(os.path.join(RepoPaths.configs_dir(), args.cfg))

    if args.dataset == "coco":
        dataset = CocoDataLoader(CocoPaths.images_dir(), CocoPaths.ids_file(), category_agnostic=False)

    elif args.dataset == "mapillary":
        dataset = MapillaryDataLoader(
            MapillaryPaths.images_dir(), MapillaryPaths.ids_file()
        )

    elif args.dataset == "pascalvoc":
        dataset = PascalVOCDataLoader(
            PascalVOCPaths.images_dir(), PascalVOCPaths.ids_file(),
            category_agnostic=False
        )
    elif args.dataset == "ytvis":
        dataset = YoutubeVISDataLoader(
            YoutubeVISPaths.training_base_dir(),
            YoutubeVISPaths.train_vds_file(),
            cfg.TRAINING.TRACKER.MAX_ITERATIONS,
            category_agnostic=False,
            single_instance_duplication=cfg.DATA.YOUTUBE_VIS.SINGLE_INSTANCE_DUPLICATION
        )
    elif args.dataset == "davis":
        dataset = DavisDataLoader(
            DavisUnsupervisedPaths.trainval_base_dir(),
            DavisUnsupervisedPaths.train_vds_file(),
            apply_augmentation=False,
            samples_to_create=cfg.DATA.DAVIS.TRAINING_SUBSEQUENCES,
            single_instance_duplication=cfg.DATA.DAVIS.SINGLE_INSTANCE_DUPLICATION
        )
    elif args.dataset == "kittimots":
        dataset = MOTSDataLoader(
            KITTIMOTSPaths.train_images_dir(), KITTIMOTSPaths.train_vds_file(),
            samples_to_create=cfg.TRAINING.TRACKER.MAX_ITERATIONS,
            apply_augmentation=cfg.DATA.KITTI_MOTS.AUGMENTATION,
            frame_gap_lower=cfg.DATA.KITTI_MOTS.FRAME_GAP_LOWER,
            frame_gap_upper=cfg.DATA.KITTI_MOTS.FRAME_GAP_UPPER
        )
    else:
        raise ValueError("Invalid dataset name given")

    visualize_data_loader_output(dataset, args.num_workers, args.batch_size, args.shuffle)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--cfg', required=True)

    parser.add_argument('--dataset', '-d', required=True)

    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', '-bS', type=int, default=1)

    main(parser.parse_args())
