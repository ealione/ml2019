import time
import numpy as np

import pandas as pd

import torchvision
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix

from architecture import *
from utils import load_whitened_dataset, AverageMeter, RecorderMeter

np.seterr(divide='ignore', invalid='ignore')


class CIFAR10Dataset(Dataset):
    def __init__(self, directory, train=True, transform=None):
        self.data, self.labels = load_whitened_dataset(directory, train)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.data[idx])
        else:
            sample = self.data[idx]
        return sample, self.labels[idx]


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
    # transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_preproc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_dataloader(preprocessed=True, train=True, transform=None, batch_size=64, num_workers=2):
    if preprocessed:
        dataset = CIFAR10Dataset(directory='data/cifar10_gcn_zca_v2.npz', train=train, transform=transform_preproc)
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)


def accuracy(output, target, topk=(1,)):
    """Computes the precision for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision_recall(output, target):
    ps = torch.exp(output)
    _, pred = ps.topk(1, dim=1)
    cm = confusion_matrix(target.cpu(), pred.cpu())
    recall_class = np.diag(cm) / np.sum(cm, axis=1)
    precision_class = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.mean(recall_class)
    precision = np.mean(precision_class)
    return recall, precision


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, model, criterion, optimizer, epoch):
    log_frequency = 200

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_frequency == 0:

            print(f"Epoch [{epoch:03d}][{i:03d}/{len(train_loader):03d}].. "
                  f"Time (batch): {batch_time.val:.3f} ({batch_time.avg:.3f}).. "
                  f"Time (data): {data_time.val:.3f} ({data_time.avg:.3f}).. "
                  f"Train loss: {losses.val:.4f} ({losses.avg:.4f}).. "
                  f"Train accuracy (top 1): {top1.val:.3f} ({top1.avg:.3f}).. "
                  f"Train accuracy (top 5): {top5.val:.3f} ({top5.avg:.3f})")

    precision, recall = precision_recall(output.data, target)
    stats = {'loss': losses.avg, 'accuracy1': top1.avg, 'accuracy5': top5.avg, 'precision': precision, 'recall': recall}

    return top1.avg, losses.avg, stats


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    precision, recall = precision_recall(output.data, target)

    print(f"  -- Test loss: {losses.val:.4f} ({losses.avg:.4f}).. "
          f"Test accuracy (top 1): {top1.val:.3f} ({top1.avg:.3f}).. "
          f"Test accuracy (top 5): {top5.val:.3f} ({top5.avg:.3f}).. "
          f"Test error: {100 - top1.avg:.3f}).. "
          f"Test precision: {precision:.3f}.. "
          f"Test recall: {recall:.3f}")

    precision, recall = precision_recall(output.data, target)
    stats = {'loss': losses.avg, 'accuracy1': top1.avg, 'accuracy5': top5.avg, 'precision': precision, 'recall': recall}

    return top1.avg, losses.avg, stats


if __name__ == "__main__":
    epochs = 50
    batch_size = 64
    learning_rate = 0.1
    momentum = 0.9
    decay = 0.002
    schedule = [100, 190, 306, 390, 440, 540]  # Decrease learning rate at these epochs
    gammas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    num_classes = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(preprocessed=True, train=True, transform=transform_train)
    test_dataloader = get_dataloader(preprocessed=True, train=False, transform=transform_test)

    model = simplenet().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005, nesterov=False)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.9, eps=1e-3,  # momentum=state['momentum'],
                                     weight_decay=0.001)

    milestones = [100, 190, 306, 390, 440, 540]
    scheduler = lr_scheduler.MultiStepLR(optimizer, schedule, gamma=0.1)

    recorder = RecorderMeter(epochs)

    stats = pd.DataFrame(
        columns=['Epoch', 'Time per epoch', 'Train loss', 'Train accuracy', 'Train top-5 accuracy', 'Train precision',
                 'Train recall', 'Test loss', 'Test accuracy', 'Test top-5 accuracy', 'Test precision', 'Test recall',
                 'Best'])

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(epochs):
        current_learning_rate = float(scheduler.get_lr()[-1])

        train_acc, train_los, t_stats = train(train_dataloader, model, criterion, optimizer, epoch)

        val_acc, val_los, v_stats = validate(test_dataloader, model, criterion)
        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        scheduler.step(epoch)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        stats = stats.append(
            {'Epoch': epoch, 'Time per epoch': epoch_time.val, 'Train loss': t_stats['loss'],
             'Train accuracy': t_stats['accuracy1'], 'Train top-5 accuracy': t_stats['accuracy5'],
             'Train precision': t_stats['precision'], 'Train recall': t_stats['recall'], 'Test loss': v_stats['loss'],
             'Test accuracy': v_stats['accuracy1'], 'Test top-5 accuracy': v_stats['accuracy5'],
             'Test precision': v_stats['precision'], 'Test recall': v_stats['recall'], 'Best': is_best},
            ignore_index=True)
    stats.to_csv(f'logs/simplenet.csv')
