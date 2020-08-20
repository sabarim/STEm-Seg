from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

from stemseg.config import cfg

from stemseg.inference.output_utils import YoutubeVISOutputGenerator, DavisOutputGenerator, KittiMOTSOutputGenerator
from stemseg.inference.online_chainer import OnlineChainer
from stemseg.inference.clusterers import SequentialClustering

from stemseg.data.generic_video_dataset_parser import parse_generic_video_dataset
from stemseg.data import DavisUnsupervisedPaths as DavisPaths, YoutubeVISPaths, KITTIMOTSPaths

from stemseg.modeling.inference_model import InferenceModel
from stemseg.modeling.embedding_utils import get_nb_free_dims

from stemseg.utils import Timer, RepoPaths

import os
import torch


def get_subsequence_frames(seq_len, subseq_len, dataset_name, frame_overlap=-1):
    subseq_idxes = []

    if dataset_name == "davis":
        frame_overlap = cfg.DATA.DAVIS.INFERENCE_FRAME_OVERLAP if frame_overlap <= 0 else frame_overlap
    elif dataset_name == "ytvis":
        frame_overlap = cfg.DATA.YOUTUBE_VIS.INFERENCE_FRAME_OVERLAP if frame_overlap <= 0 else frame_overlap
    elif dataset_name == "kittimots":
        frame_overlap = cfg.DATA.KITTI_MOTS.INFERENCE_FRAME_OVERLAP if frame_overlap <= 0 else frame_overlap
    else:
        raise NotImplementedError()

    assert frame_overlap < subseq_len

    if seq_len < subseq_len:
        padded_frames = [True for _ in range(subseq_len - seq_len)] + [False for _ in range(seq_len)]
        return [[0 for _ in range(subseq_len - seq_len)] + list(range(seq_len))], padded_frames

    last_frame_idx = -1
    for t in range(0, seq_len - subseq_len + 1, subseq_len - frame_overlap):
        subseq_idxes.append(list(range(t, t + subseq_len)))
        last_frame_idx = subseq_idxes[-1][-1]

    if last_frame_idx != seq_len - 1:
        subseq_idxes.append(list(range(seq_len - subseq_len, seq_len)))

    return subseq_idxes, None


class TrackGenerator(object):
    def __init__(self, sequences, dataset_name, output_generator, output_dir, model_ckpt_path, max_tracks, preload_images,
                 resize_scale, semseg_averaging_on_gpu, **kwargs):
        self.sequences = sequences
        self.dataset_name = dataset_name

        self.output_generator = output_generator
        if self.dataset_name == "kittimots":
            semseg_output_type = "argmax"
        elif self.dataset_name == "ytvis":
            semseg_output_type = "logits"
        else:
            semseg_output_type = None

        self.model = InferenceModel(model_ckpt_path, semseg_output_type=semseg_output_type, preload_images=preload_images,
                                    resize_scale=resize_scale, semseg_generation_on_gpu=semseg_averaging_on_gpu).cuda()

        self.resize_scale = resize_scale
        self.vis_output_dir = os.path.join(output_dir, "vis")
        self.embeddings_output_dir = os.path.join(output_dir, "embeddings")
        self.max_tracks = max_tracks

        self.save_vis = kwargs.get("save_vis", False)

        self.seediness_fg_threshold = kwargs.get("seediness_thresh", 0.25)
        self.ignore_fg_masks = kwargs.get("ignore_fg_masks", False)
        self.frame_overlap = kwargs.get("frame_overlap", -1)
        self.clustering_device = kwargs.get("clustering_device", "cuda:0")

        self.chainer = OnlineChainer(self.create_clusterer(), embedding_resize_factor=resize_scale)
        self.total_frames_processed = 0.

    def create_clusterer(self):
        _cfg = cfg.CLUSTERING
        return SequentialClustering(primary_prob_thresh=_cfg.PRIMARY_PROB_THRESHOLD,
                                    secondary_prob_thresh=_cfg.SECONDARY_PROB_THRESHOLD,
                                    min_seediness_prob=_cfg.MIN_SEEDINESS_PROB,
                                    n_free_dims=get_nb_free_dims(cfg.MODEL.EMBEDDING_DIM_MODE),
                                    free_dim_stds=cfg.TRAINING.LOSSES.EMBEDDING.FREE_DIM_STDS,
                                    device=self.clustering_device)

    def get_fg_masks_from_seediness(self, inference_output):
        seediness_scores = defaultdict(lambda: [0., 0.])

        for subseq_frames, _, _, subseq_seediness in inference_output['embeddings']:
            subseq_seediness = subseq_seediness.cuda().squeeze(0)
            for i, t in enumerate(subseq_frames):
                seediness_scores[t][0] += subseq_seediness[i]
                seediness_scores[t][1] += 1.

        fg_masks = [(seediness_scores[t][0] / seediness_scores[t][1]) for t in sorted(seediness_scores.keys())]
        return (torch.stack(fg_masks, 0) > self.seediness_fg_threshold).byte().cpu()

    def start(self, seqs_to_process):
        # iou_stats_container = IoUStatisticsContainer()

        if not isinstance(self.max_tracks, (list, tuple)):
            self.max_tracks = [self.max_tracks] * len(self.sequences)

        for i in range(len(self.sequences)):
            sequence = self.sequences[i]
            if seqs_to_process and str(sequence.seq_id) not in seqs_to_process:
                continue

            print("Performing inference for sequence {}/{}".format(i + 1, len(self.sequences)))
            self.process_sequence(sequence, self.max_tracks[i])

        print("----------------------------------------------------")
        print("Model inference speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_duration("inference")))
        print("Clustering and postprocessing speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_duration("postprocessing")))
        print("Overall speed: {:.3f} fps".format(self.total_frames_processed / Timer.get_durations_sum()))
        print("----------------------------------------------------")

    def process_sequence(self, sequence, max_tracks):
        embeddings, fg_masks, multiclass_masks = self.do_inference(sequence)

        self.do_clustering(sequence, embeddings, fg_masks, multiclass_masks, max_tracks)

        self.total_frames_processed += len(sequence)

    @Timer.log_duration("inference")
    def do_inference(self, sequence):
        subseq_idxes, _ = get_subsequence_frames(
            len(sequence), cfg.INPUT.NUM_FRAMES, self.dataset_name, self.frame_overlap)

        image_paths = [os.path.join(sequence.base_dir, path) for path in sequence.image_paths]
        inference_output = self.model(image_paths, subseq_idxes)

        fg_masks, multiclass_masks = inference_output['fg_masks'], inference_output['multiclass_masks']

        if torch.is_tensor(fg_masks):
            print("Obtaining foreground mask from model's foreground mask output")
            fg_masks = (fg_masks > 0.5).byte()  # [T, H, W]
        else:
            print("Obtaining foreground mask by thresholding seediness map at {}".format(self.seediness_fg_threshold))
            fg_masks = self.get_fg_masks_from_seediness(inference_output)

        return inference_output["embeddings"], fg_masks, multiclass_masks

    @Timer.log_duration("postprocessing")
    def do_clustering(self, sequence, all_embeddings, fg_masks, multiclass_masks, max_tracks):
        subseq_dicts = []

        for i, (subseq_frames, embeddings, bandwidths, seediness) in tqdm(enumerate(all_embeddings), total=len(all_embeddings)):
            subseq_dicts.append({
                "frames": subseq_frames,
                "embeddings": embeddings,
                "bandwidths": bandwidths,
                "seediness": seediness,
            })

        (track_labels, instance_pt_counts, instance_lifetimes), framewise_mask_idxes, subseq_labels_list, \
            fg_embeddings, subseq_clustering_meta_info = self.chainer.process(
            fg_masks, subseq_dicts)

        self.output_generator.process_sequence(
            sequence, framewise_mask_idxes, track_labels, instance_pt_counts, instance_lifetimes, multiclass_masks,
            fg_masks.shape[-2:], 4.0, max_tracks, device=self.clustering_device
        )


def configure_directories(args):
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(args.model_path), "inference")

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(args.model_path), output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_cfg(args):
    cfg_file = os.path.join(os.path.dirname(args.model_path), 'config.yaml')
    if not os.path.exists(cfg_file):
        dataset_cfgs = {
            "davis": "davis_2.yaml",
            "ytvis": "youtube_vis.yaml",
            "kittimots": "kitti_mots_2.yaml"
        }
        assert args.dataset in dataset_cfgs, \
            "Invalid '--dataset' argument. Should be either 'davis', 'ytvis' or 'kittimots'"
        cfg_file = os.path.join(RepoPaths.configs_dir(), dataset_cfgs[args.dataset])

    print("Loading config from {}".format(cfg_file))
    cfg.merge_from_file(cfg_file)


def configure_input_dims(args):
    if not args.min_dim and not args.max_dim:
        return

    elif args.min_dim and args.max_dim:
        assert args.min_dim > 0
        assert args.max_dim > 0
        cfg.INPUT.update_param("MIN_DIM", args.min_dim)
        cfg.INPUT.update_param("MAX_DIM", args.max_dim)

    elif args.min_dim and not args.max_dim:
        assert args.min_dim > 0
        dim_ratio = float(cfg.INPUT.MAX_DIM) / float(cfg.INPUT.MIN_DIM)
        cfg.INPUT.update_param("MIN_DIM", args.min_dim)
        cfg.INPUT.update_param("MAX_DIM", int(round(args.min_dim * dim_ratio)))

    elif not args.min_dim and args.max_dim:
        assert args.max_dim > 0
        dim_ratio = float(cfg.INPUT.MAX_DIM) / float(cfg.INPUT.MIN_DIM)
        cfg.INPUT.update_param("MIN_DIM", int(round(args.max_dim / dim_ratio)))
        cfg.INPUT.update_param("MAX_DIM", args.max_dim)

    else:
        raise ValueError("Should never be here")

    print("Network input image dimension limits: {}, {}".format(cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM))


def main(args):
    # update cfg according to backbone choice
    load_cfg(args)

    if args.min_seediness_prob:
        print("Min seediness prob for instance center --> {}".format(args.min_seediness_prob))
        cfg.CLUSTERING.update_param("MIN_SEEDINESS_PROB", args.min_seediness_prob)

    configure_input_dims(args)

    output_dir = configure_directories(args)
    preload_images = True

    cluster_full_scale = cfg.TRAINING.LOSS_AT_FULL_RES or args.resize_embeddings
    output_resize_scale = 4.0 if cluster_full_scale else 1.0
    semseg_averaging_on_gpu = not ((args.dataset == "ytvis" or args.dataset == "kittimots") and cluster_full_scale)

    if args.dataset == "davis":
        sequences, _ = parse_generic_video_dataset(DavisPaths.trainval_base_dir(), DavisPaths.val_vds_file())
        output_generator = DavisOutputGenerator(output_dir, OnlineChainer.OUTLIER_LABEL, args.save_vis,
                                                upscaled_inputs=cluster_full_scale)
        max_tracks = cfg.DATA.DAVIS.MAX_INFERENCE_TRACKS

    elif args.dataset in "ytvis":
        sequences, meta_info = parse_generic_video_dataset(YoutubeVISPaths.val_base_dir(),
                                                           YoutubeVISPaths.val_vds_file())
        output_generator = YoutubeVISOutputGenerator(output_dir, OnlineChainer.OUTLIER_LABEL, args.save_vis,
                                                     None, meta_info["category_labels"],
                                                     upscaled_inputs=cluster_full_scale)
        max_tracks = cfg.DATA.YOUTUBE_VIS.MAX_INFERENCE_TRACKS

    elif args.dataset == "kittimots":
        sequences, _ = parse_generic_video_dataset(KITTIMOTSPaths.train_images_dir(), KITTIMOTSPaths.val_vds_file())
        output_generator = KittiMOTSOutputGenerator(output_dir, OnlineChainer.OUTLIER_LABEL, args.save_vis,
                                                    upscaled_inputs=cluster_full_scale)
        max_tracks = cfg.DATA.KITTI_MOTS.MAX_INFERENCE_TRACKS
        preload_images = False

    else:
        raise ValueError("Invalid dataset name {} provided".format(args.dataset))

    max_tracks = args.max_tracks if args.max_tracks else max_tracks

    track_generator = TrackGenerator(
        sequences, args.dataset, output_generator, output_dir, args.model_path,
        save_vis=args.save_vis,
        seediness_thresh=args.seediness_thresh,
        frame_overlap=args.frame_overlap,
        max_tracks=max_tracks,
        preload_images=preload_images,
        resize_scale=output_resize_scale,
        semseg_averaging_on_gpu=semseg_averaging_on_gpu,
        clustering_device=args.clustering_device
    )

    track_generator.start(args.seqs)
    output_generator.save()
    print("Results saved to {}".format(output_dir))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path')

    parser.add_argument('--output_dir', '-o',              required=False)
    parser.add_argument('--seqs',       nargs="*",         required=False)
    parser.add_argument('--dataset',    '-d',              required=True)

    parser.add_argument('--max_tracks',           type=int,      required=False)
    parser.add_argument('--frame_overlap', '-fo', type=int,         default=-1)
    parser.add_argument('--seediness_thresh', '-st', type=float,    default=0.25)

    parser.add_argument('--min_dim', type=int, required=False)
    parser.add_argument('--max_dim', type=int, required=False)

    parser.add_argument('--resize_embeddings',  action='store_true')
    parser.add_argument('--min_seediness_prob', '-msp', type=float, required=False)
    parser.add_argument('--clustering_device', default="cuda:0")

    parser.add_argument('--save_vis',      action='store_true')

    main(parser.parse_args())
