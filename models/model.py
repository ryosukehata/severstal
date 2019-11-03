import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import sys
import os

deeplabv3plus_PATH = 'DEEPLABV3PLUS_PATH'


def kaeru_classify_model(num_classes=4):
    '''
    resnet34 custom model for classification whather defect has,
    based on Heng CherKeng models
    https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106462#latest-653034
    '''
    model = torchvision.models.resnet34()
    model.avgpool = nn.Sequential(
        nn.Dropout(),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(512, 32, kernel_size=1),
        nn.Conv2d(32, num_classes, kernel_size=1),
        nn.Identity()
    )

    model.fc = nn.Sequential(
        nn.Identity()
    )
    return model


def create_segmentation_models(encoder, arch, num_classes=4,
                               encoder_weights=None, activation=None):
    '''
    segmentation_models_pytorch https://github.com/qubvel/segmentation_models.pytorch
    has following architectures: 
    - Unet
    - Linknet
    - FPN
    - PSPNet
    encoders: A lot! see the above github page.

    Deeplabv3+ https://github.com/jfzhang95/pytorch-deeplab-xception
    has for encoders:
    - resnet (resnet101)
    - mobilenet 
    - xception
    - drn
    '''
    if arch == "Unet":
        return smp.Unet(encoder, encoder_weights=encoder_weights,
                        classes=num_classes, activation=activation)
    elif arch == "Linknet":
        return smp.Linknet(encoder, encoder_weights=encoder_weghts,
                           classes=num_classes, activation=activation)
    elif arch == "FPN":
        return smp.FPN(encoder, encoder_weights=encoder_weghts,
                       classes=num_classes, activation=activation)
    elif arch == "PSPNet":
        return smp.PSPNet(encoder, encoder_weights=encoder_weghts,
                          classes=num_classes, activation=activation)
    elif arch == "deeplabv3plus":
        if deeplabv3plus_PATH in os.environ:
            sys.path.append(os.environ[deeplabv3plus_PATH])
            from modeling.deeplab import DeepLab
            return DeepLab(encoder, num_classes=4)
        else:
            raise ValueError('Set deeplabv3plus path by environment variable.')
    else:
        raise ValueError('arch {} is not found, set the correct arch'.format(arch))
        sys.exit()


if __name__ == '__main__':
    model = create_segmentation_models("resnet34", "Unet")
    print(model)
