
import os
from functools import partial

import torch
from torch.nn.modules import Module

import torchvision
from torchvision.models._api import WeightsEnum
from torchvision.models import resnet as R
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

try:
    from torchinfo import summary
except ImportError:
    torchinfo_summary = None

import segmentation_models_pytorch as smp
import timm

from components import FCN
from models.utils import extract_backbone, load_state_dict, modify_resnet
# from models.components import get_weight


BACKBONE_LAT_DIM_MAP = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnext50_32x4d': 2048,
    'resnext101_32x8d': 2048,
    'wide_resnet50_2': 2048,
    'wide_resnet101_2': 2048,
}

BACKBONE_WEIGHT_MAP = {
    'resnet18': R.ResNet18_Weights.DEFAULT,
    'resnet34': R.ResNet34_Weights.DEFAULT,
    'resnet50': R.ResNet50_Weights.DEFAULT,
    'resnet101': R.ResNet101_Weights.DEFAULT,
    'resnet152': R.ResNet152_Weights.DEFAULT,
    'resnext50_32x4d': R.ResNeXt50_32X4D_Weights.DEFAULT,
    'resnext101_32x8d': R.ResNeXt101_32X8D_Weights.DEFAULT,
    'wide_resnet50_2': R.Wide_ResNet50_2_Weights.DEFAULT,
    'wide_resnet101_2': R.Wide_ResNet101_2_Weights.DEFAULT,
}

def SemanticSegmentationModel(model: str ="unet",
                              backbone: str ="resnet50",
                              weights: WeightsEnum | str | bool | None = None,
                              in_channels: int =3,
                              num_classes: int =1000,
                              num_filters: int = 3,
                              freeze_backbone: bool = False,
                              freeze_decoder: bool = False, ) -> Module | None:
    """
    Initialize a new SemanticSegmentation model.

    Args:
        model: Name of the
            `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
        backbone: Name of the `timm
            <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
            <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
        weights: Initial model weights. Either a weight enum, the string
            representation of a weight enum, True for ImageNet weights, False or
            None for random weights, or the path to a saved model state dict. FCN
            model does not support pretrained weights. Pretrained ViT weight enums
            are not supported yet.
        in_channels: Number of input channels to model.
        num_classes: Number of prediction classes (including the background).
        num_filters: Number of filters. Only applicable when model='fcn'.
        freeze_backbone: Freeze the backbone network to fine-tune the
            decoder and segmentation head.
        freeze_decoder: Freeze the decoder network to linear probe
            the segmentation head.

    .. versionchanged:: 0.3
        *ignore_zeros* was renamed to *ignore_index*.

    .. versionchanged:: 0.4
        *segmentation_model*, *encoder_name*, and *encoder_weights*
        were renamed to *model*, *backbone*, and *weights*.

    .. versionadded:: 0.5
        The *class_weights*, *freeze_backbone*, and *freeze_decoder* parameters.

    .. versionchanged:: 0.5
        The *weights* parameter now supports WeightEnums and checkpoint paths.
        *learning_rate* and *learning_rate_schedule_patience* were renamed to
        *lr* and *patience*.

    .. versionchanged:: 0.6
        The *ignore_index* parameter now works for jaccard loss.

    Raises:
        ValueError: If *model* is invalid.
    """

    if model == 'unet':
        seg_model = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if weights is True else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model == 'deeplabv3+':
        seg_model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights='imagenet' if weights is True else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model == 'fcn':
        seg_model = FCN(
            in_channels=in_channels, classes=num_classes, num_filters=num_filters
        )
    else:
        raise ValueError(
            f"Model type '{model}' is not valid. "
            "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
        )

    if model != 'fcn':
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = extract_backbone(weights)
            else:
                pass
                # state_dict = get_weight(weights).get_state_dict(progress=True)
            seg_model.encoder.load_state_dict(state_dict)

    # Freeze backbone
    if freeze_backbone and model in ['unet', 'deeplabv3+']:
        for param in seg_model.encoder.parameters():
            param.requires_grad = False

    # Freeze decoder
    if freeze_decoder and model in ['unet', 'deeplabv3+']:
        for param in seg_model.decoder.parameters():
            param.requires_grad = False

    return seg_model


def ClassificationModel(model: str = 'resnet50',
                        weights: WeightsEnum | str | bool | None = None,
                        in_channels: int = 3,
                        num_classes: int = 1000,
                        freeze_backbone: bool = False) -> Module | None:
    """
    Initialize a new Classification model.

    Args:
        model: Name of the `timm
            <https://huggingface.co/docs/timm/reference/models>`__ model to use.
        weights: Initial model weights. Either a weight enum, the string
            representation of a weight enum, True for ImageNet weights, False
            or None for random weights, or the path to a saved model state dict.
        in_channels: Number of input channels to model.
        num_classes: Number of prediction classes.
        freeze_backbone: Freeze the backbone network to linear probe
            the classifier head.

    .. versionchanged:: 0.4
       *classification_model* was renamed to *model*.

    .. versionadded:: 0.5
       The *class_weights* and *freeze_backbone* parameters.

    .. versionchanged:: 0.5
       *learning_rate* and *learning_rate_schedule_patience* were renamed to
       *lr* and *patience*.
    """

    # Create model
    cls_model = timm.create_model(
        model,
        num_classes=num_classes,
        in_chans=in_channels,
        pretrained=weights is True,
    )

    # Load weights
    if weights and weights is not True:
        if isinstance(weights, WeightsEnum):
            state_dict = weights.get_state_dict(progress=True)
        elif os.path.exists(weights):
            _, state_dict = extract_backbone(weights)
        else:
            pass
            # state_dict = get_weight(weights).get_state_dict(progress=True)
        load_state_dict(cls_model, state_dict)

    # Freeze backbone and unfreeze classifier head
    if freeze_backbone:
        for param in cls_model.parameters():
            param.requires_grad = False
        for param in cls_model.get_classifier().parameters():
            param.requires_grad = True
    return cls_model


def ObjectDetectionModel(model: str = 'faster-rcnn',
                         backbone: str = 'resnet50',
                         weights: bool | None = None,
                         num_classes: int = 1000,
                         trainable_layers: int = 3,
                         freeze_backbone: bool = False,) -> Module | None:
    """"
    Initialize a new ObjectDetection model.

    Args:
        model: Name of the `torchvision
            <https://pytorch.org/vision/stable/models.html#object-detection>`__
            model to use. One of 'faster-rcnn', 'fcos', or 'retinanet'.
        backbone: Name of the `torchvision
            <https://pytorch.org/vision/stable/models.html#classification>`__
            backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
            'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'wide_resnet50_2', or 'wide_resnet101_2'.
        weights: Initial model weights. True for ImageNet weights, False or None
            for random weights.
        num_classes: Number of prediction classes (including the background).
        trainable_layers: Number of trainable layers.
        freeze_backbone: Freeze the backbone network to fine-tune the detection head.

    Raises:
        ValueError: If *model* or *backbone* are invalid.
    """

    if backbone in BACKBONE_LAT_DIM_MAP:
        kwargs = {
            'backbone_name': backbone,
            'trainable_layers': trainable_layers
        }
        if weights:
            kwargs['weights'] = BACKBONE_WEIGHT_MAP[backbone]
        else:
            kwargs['weights'] = None

        latent_dim = BACKBONE_LAT_DIM_MAP[backbone]
    else:
        raise ValueError(f"Backbone type '{backbone}' is not valid.")

    if model == 'faster-rcnn':
        model_backbone = resnet_fpn_backbone(**kwargs)
        anchor_generator = AnchorGenerator(
            sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
        )

        if freeze_backbone:
            for param in model_backbone.parameters():
                param.requires_grad = False

        det_model = torchvision.models.detection.FasterRCNN(
            model_backbone,
            num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif model == 'fcos':
        kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(256, 256)
        kwargs['norm_layer'] = (
            misc.FrozenBatchNorm2d if weights else torch.nn.BatchNorm2d
        )

        model_backbone = resnet_fpn_backbone(**kwargs)
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
        )

        if freeze_backbone:
            for param in model_backbone.parameters():
                param.requires_grad = False

        det_model = torchvision.models.detection.FCOS(
            model_backbone, num_classes, anchor_generator=anchor_generator
        )
    elif model == 'retinanet':
        kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(
            latent_dim, 256
        )
        model_backbone = resnet_fpn_backbone(**kwargs)

        anchor_sizes = (
            (16, 20, 25),
            (32, 40, 50),
            (64, 80, 101),
            (128, 161, 203),
            (256, 322, 406),
            (512, 645, 812),
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        head = RetinaNetHead(
            model_backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )

        if freeze_backbone:
            for param in model_backbone.parameters():
                param.requires_grad = False

        det_model = torchvision.models.detection.RetinaNet(
            model_backbone,
            num_classes,
            anchor_generator=anchor_generator,
            head=head,
        )
    else:
        raise ValueError(f"Model type '{model}' is not valid.")
    return det_model


if __name__ == '__main__':
    model = SemanticSegmentationModel(model='unet', backbone='resnet50', num_classes=10)
    # model = ClassificationModel(model='resnet50', weights=False)
    batch_size = 1
    input_size = (batch_size, 3, 256, 256)
    input_data = torch.randn(input_size)
    output = model(input_data)
    print(output.shape)
    # summary(model, input_size=(batch_size, 3, 256, 256))
