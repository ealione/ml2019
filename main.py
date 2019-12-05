import time
import numpy as np

import pandas as pd

import torchvision
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix

from architecture.optimized import *
from utils import load_whitened_dataset


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
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
    # transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_dataloader(preprocessed=True, train=True, transform=None):
    if preprocessed:
        dataset = CIFAR10Dataset(directory='data/cifar10_gcn_zca_v2.npz', train=train, transform=transform_test)
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


def run(trainloader, testloader, model, criterion, optimizer, epochs):
    stats = pd.DataFrame(
        columns=['Epoch', 'Time per epoch', 'Train loss', 'Train accuracy', 'Train top-3 accuracy', 'Test loss',
                 'Test accuracy', 'Test top-3 accuracy'])

    steps = 0
    running_loss = 0
    model.train()
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

                cm = confusion_matrix(labels, top_class)
                recall = np.diag(cm) / np.sum(cm, axis=1)
                precision = np.diag(cm) / np.sum(cm, axis=0)

        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Time per epoch: {time_elapsed:.4f}.. "
              f"Train loss: {running_loss / len(trainloader):.4f}.. "
              f"Train accuracy: {train_accuracy / len(trainloader):.4f}.. "
              f"Top-3 train accuracy: {top3_train_accuracy / len(trainloader):.4f}",
              f"Test loss: {test_loss / len(testloader):.4f}.. "
              f"Test accuracy: {test_accuracy / len(testloader):.4f}.. "
              f"Top-3 test accuracy: {top3_test_accuracy / len(testloader):.4f}"
              f"Test precision: {precision:.4f}.. "
              f"Test recall: {recall:.4f}")

        stats = stats.append(
            {'Epoch': epoch, 'Time per epoch': time_elapsed, 'Train loss': running_loss / len(trainloader),
             'Train accuracy': train_accuracy / len(trainloader),
             'Train top-3 accuracy': top3_train_accuracy / len(trainloader), 'Test loss': test_loss / len(testloader),
             'Test accuracy': test_accuracy / len(testloader),
             'Test top-3 accuracy': top3_test_accuracy / len(testloader), 'Test precision': precision,
             'Test recall': recall}, ignore_index=True)

        running_loss = 0
        model.train()
    return stats


if __name__ == "__main__":
    epochs = 100
    batch_size = 128
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(preprocessed=False, train=True, transform=transform_train)
    test_dataloader = get_dataloader(preprocessed=False, train=False, transform=transform_test)

    model = mobilenetv2(activation='mish').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_stats = pd.DataFrame(
        columns=['Epoch', 'Time per epoch', 'Train loss', 'Train accuracy', 'Train top-3 accuracy', 'Test loss',
                 'Test accuracy', 'Test top-3 accuracy'])

    stats = run(train_dataloader, test_dataloader, model, criterion, optimizer, epochs)
    stats.to_csv('logs/test.csv')
