import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    """
    单个 epoch 的训练过程。
    """

    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    loop = tqdm(train_loader, desc=f"Train Epoch [{epoch}/{epochs}]", leave=False)

    for images, targets in loop:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        acc = accuracy(outputs, targets)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        loop.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}%"
        })

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型。
    """

    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        acc = accuracy(outputs, targets)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


def get_current_lr(optimizer):
    """
    获取当前学习率。
    """
    return optimizer.param_groups[0]["lr"]