import time
import numpy as np

from torch.optim import Adam

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from architecture.optimized import *

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_training_dataloader(train_transform, batch_size=128, num_workers=0, shuffle=True):
    transform_train = train_transform
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader


def get_testing_dataloader(test_transform, batch_size=128, num_workers=0, shuffle=True):
    transform_test = test_transform
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader


def train(model, criterion, optimizer, epochs):
    model.train()

    steps = 0
    running_loss = 0
    for epoch in range(epochs):

        since = time.time()

        train_accuracy = 0
        top3_train_accuracy = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_train_accuracy += np.mean(
                [1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        time_elapsed = time.time() - since

        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Time per epoch: {time_elapsed:.4f}.. "
              f"Average time per step: {time_elapsed / len(trainloader):.4f}.. "
              f"Train loss: {running_loss / len(trainloader):.4f}.. "
              f"Train accuracy: {train_accuracy / len(trainloader):.4f}.. "
              f"Top-3 train accuracy: {top3_train_accuracy / len(trainloader):.4f}")

        running_loss = 0


def test(model, criterion, epochs):
    for epoch in range(epochs):
        test_loss = 0
        test_accuracy = 0
        top3_test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
                target_numpy = labels.cpu().numpy()
                top3_test_accuracy += np.mean(
                    [1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Test loss: {test_loss / len(testloader):.4f}.. "
              f"Test accuracy: {test_accuracy / len(testloader):.4f}.. "
              f"Top-3 test accuracy: {top3_test_accuracy / len(testloader):.4f}")


if __name__ == "__main__":
    epochs = 100
    batch_size = 128
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    trainloader = get_training_dataloader(train_transform)
    testloader = get_testing_dataloader(test_transform)

    model = mobilenetv2(activation='mish').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, epochs)
    test(model, criterion, epochs)
