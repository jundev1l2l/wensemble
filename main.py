import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main(args):
    os.makedirs(f"result", exist_ok=True)
    file_handler = logging.FileHandler(f"result/{args.exp}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    dataset = get_dataset()
    logger.info("Train Dataset Config")
    logger.info("-" * 30)
    logger.info(dataset["train"])
    logger.info(dataset["train"].data.size())
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset["train"]), size=(1,)).item()
        img, label = dataset["train"][sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    plt.close()
    logger.info("")
    logger.info("Test Dataset Config")
    logger.info("-" * 30)
    logger.info(dataset["test"])
    logger.info(dataset["test"].data.size())
    logger.info("")

    dataloader = get_dataloader(dataset)

    model = CNN()
    logger.info("Model Config")
    logger.info("-" * 30)
    logger.info(model)
    logger.info("")

    loss_func = nn.CrossEntropyLoss()
    logger.info("Loss Function Config")
    logger.info("-" * 30)
    logger.info(loss_func)
    logger.info("")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    logger.info("Optimizer Config")
    logger.info("-" * 30)
    logger.info(optimizer)
    logger.info("")

    train(dataloader, model, loss_func, optimizer)


def train(dataloader, model, loss_func, optimizer, num_epoch=100, print_epoch=1):
    logger.info("Train Started ...")
    logger.info("-" * 30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    num_batch = len(dataloader)
    for epoch in range(num_epoch):
        loss = 0

        for batch_idx, (x, y) in enumerate(dataloader["train"]):
            logits, _ = model(x.to(device))
            loss_batch = loss_func(logits, y.to(device))
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss += loss_batch
        loss /= num_batch

        if (epoch + 1) % print_epoch == 0:
            logger.info(f"Epoch [{epoch+1:03d}/{num_epoch:03d}, Loss: {loss.item():.4f}]")

    torch.save(model.state_dict(), f"~/../../data/junhyun/ckpt/{args.exp}.tar")



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def get_dataloader(dataset):

    train_loader = DataLoader(dataset["train"],
                              batch_size=100,
                              shuffle=True,
                              num_workers=1)

    test_loader = DataLoader(dataset["test"],
                             batch_size=100,
                             shuffle=True,
                             num_workers=1)

    return {"train": train_loader, "test": test_loader}


def get_dataset():

    train_data = datasets.MNIST(
        root="~/../../data/junhyun/",
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root="~/../../data/junhyun/",
        train=False,
        transform=ToTensor(),
    )

    return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "-e", type=str, default="debug")
    args = parser.parse_args()

    main(args)
