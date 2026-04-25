import torch


def get_optimizer(optimizer_name, model, lr=None, weight_decay=5e-4):
    """
    根据名称返回不同优化器。

    注意：
    - SGD 通常使用较大的学习率，例如 0.1
    - Adam / AdamW 通常使用较小学习率，例如 0.001
    """

    name = optimizer_name.lower()

    if name == "sgd":
        if lr is None:
            lr = 0.1
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "sgdm":
        if lr is None:
            lr = 0.1
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif name == "adam":
        if lr is None:
            lr = 0.001
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "adamw":
        if lr is None:
            lr = 0.001
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "rmsprop":
        if lr is None:
            lr = 0.001
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif name == "adagrad":
        if lr is None:
            lr = 0.01
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(optimizer, scheduler_name, epochs):
    """
    学习率调度器。

    为了简单，可以选择：
    - none：不使用学习率调度
    - cosine：余弦退火
    - step：分段下降
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none":
        return None

    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[epochs // 2, int(epochs * 0.75)],
            gamma=0.1
        )

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")