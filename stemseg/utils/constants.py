import os


mode_to_config_mapping = {
    "davis": "davis",
    "davis_pretraining": "davis",
    "davis_ft": "davis",
    "youtube_vis": "youtube_vis",
    "kitti_train": "kitti_mots",
    "kitti_trainval": "kitti_mots",
    "mapillary": "mapillary"
}


class Loss(object):
    EMBEDDING_VARIANCE = "embedding_variance_loss"
    EMBEDDING_DISTANCE = "embedding_distance_loss"
    EMBEDDING = "embedding_loss"
    SEMSEG = "semantic_segmentation_loss"
    AUXILIARY = "auxiliary"
    EIGENVALUE_RATIO = "eigenvalue_ratio_loss"
    LOVASZ_LOSS = "lovasz_loss"
    SEEDINESS_LOSS = "seediness_loss"
    VARIANCE_SMOOTHNESS = "variance_smoothness_loss"
    FOREGROUND = "foreground"

    def __init__(self):
        raise ValueError("Static class 'Losses' should not be instantiated")


class ModelOutput(object):
    TRACKER_INPUT_FEATURES = "tracker_input_features",
    SEMSEG_MASKS = "semseg_masks",
    EMBEDDINGS = "embeddings",
    EMBEDDING_VARIANCES = "variances"
    SEEDINESS_MAP = "seediness_map"
    EMBEDDING_OFFSETS = "embedding_offsets"
    MASK_GRADIENTS = "mask_gradients"
    OFFSET_IMPROVEMENT_FACTOR = "improvement_factor"

    INFERENCE = "inference"
    OPTIMIZATION_LOSSES = "optimization_losses"
    OTHERS = "others"

    def __init__(self):
        raise ValueError("Static class 'ModelOutput' should not be instantiated")


class RepoPaths(object):
    def __init__(self):
        raise ValueError("Static class 'RepoPaths' should not be instantiated")

    @staticmethod
    def dataset_meta_info_dir():
        return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'metainfo'))

    @staticmethod
    def configs_dir():
        return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'config'))
