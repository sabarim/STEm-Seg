import os


def _get_env_var(varname):
    value = os.getenv(varname)
    if not value:
        raise EnvironmentError("Required environment variable '{}' is not set.".format(varname))
    return value


class ModelPaths(object):
    def __init__(self):
        pass

    @staticmethod
    def checkpoint_base_dir():
        return os.path.join(_get_env_var('STEMSEG_MODELS_DIR'), 'checkpoints')

    @staticmethod
    def pretrained_backbones_dir():
        return os.path.join(_get_env_var('STEMSEG_MODELS_DIR'), 'pretrained')

    @staticmethod
    def pretrained_maskrcnn_x101_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth')

    @staticmethod
    def pretrained_maskrcnn_r50_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_R_50_FPN_1x.pth')

    @staticmethod
    def pretrained_maskrcnn_r101_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_R_101_FPN_1x.pth')