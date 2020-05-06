import argparse
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from prepare_model import MNISTNet

import config


def train(
        model: nn.Module,
        device,
        train_loader,
        optimizer,
        epoch: int,
        log_interval: int
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_loop(
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 14,
        lr: float = 1.0,
        gamma: float = 0.7,
        no_cuda: bool = True,
        seed: int = 1,
        log_interval: int = 10,
        save_model: bool = False
):
    """

    :param batch_size: input batch size for training (default: 64)
    :param test_batch_size: input batch size for testing (default: 1000)
    :param epochs: number of epochs to train (default: 14)
    :param lr: learning rate (default: 1.0)
    :param gamma: Learning rate step gamma (default: 0.7)
    :param no_cuda: disables CUDA training
    :param seed: random seed (default: 1)
    :param log_interval: how many batches to wait before logging training status
    :param save_model: For Saving the current Model
    :return:
    """

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            config.DATA_DIR, train=True, download=True,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
        ),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            config.DATA_DIR, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=test_batch_size, shuffle=True, **kwargs
    )

    model = MNISTNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), join(config.MODEL_DIR, 'mnist_cnn.pt'))
