import torch.nn as nn
import torchvision.models as models


def get_resnet18_cifar10(num_classes=10):
    """
    构建适用于 CIFAR-10 的 ResNet18。

    原版 ResNet18 适用于 ImageNet，输入尺寸通常是 224x224。
    CIFAR-10 图片是 32x32，因此需要修改：
    1. 第一层卷积改成 3x3，stride=1
    2. 去掉 maxpool
    """

    try:
        model = models.resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        # 兼容旧版本 torchvision
        model = models.resnet18(pretrained=False, num_classes=num_classes)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    return model


def get_model(model_name="resnet18", num_classes=10):
    model_name = model_name.lower()

    if model_name == "resnet18":
        return get_resnet18_cifar10(num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")