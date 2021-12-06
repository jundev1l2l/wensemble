import os
import argparse
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import StratifiedShuffleSplit


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    logger.info("Val Dataset Config")
    logger.info("-" * 30)
    logger.info(dataset["val"])
    logger.info(dataset["val"].data.size())
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


def train(dataloader, model, loss_func, optimizer, num_epoch=100, print_epoch=1, eval_epoch=5):

    logger.info("Train Started ...")
    logger.info("-" * 30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_batch_train = len(dataloader["train"])
    num_batch_val = len(dataloader["val"])

    best_acc = .0
    best_epoch = 0
    for epoch in range(num_epoch):
        model.train()
        loss = 0
        for batch_idx, (x, y) in enumerate(tqdm(dataloader["train"])):
            logits = model(x.to(device))
            loss_batch = loss_func(logits, y.to(device))
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss += loss_batch
        loss /= num_batch_train

        if (epoch + 1) % print_epoch == 0:
            logger.info(f"Epoch [{epoch+1:03d}/{num_epoch:03d}], Loss: {loss.item():.4f}")

        if (epoch + 1) % eval_epoch == 0:
            model.eval()
            loss = 0
            acc = 0
            for x, y in dataloader["val"]:
                with torch.no_grad():
                    preds, logits = model.predict(x.to(device))
                    loss_batch = loss_func(logits, y.to(device))
                    loss += loss_batch
                    acc_batch = (preds.cpu()==y).sum() / len(y)
                    acc += acc_batch
            loss /= num_batch_val
            acc /= num_batch_val

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                logger.info(f"Val_Loss: {loss.item():.4f}, Val_Acc: {acc.item():.4f} ------> Best Model Saved")
                os.makedirs(f"~/../../data/junhyun/ckpt", exist_ok=True)
                torch.save(model.state_dict(), f"~/../../data/junhyun/ckpt/{args.exp}.tar")
            else:
                logger.info(f"Val_Loss: {loss.item():.4f}, Val_Acc: {acc.item():.4f}")

    logger.info(f"")
    logger.info(f"Train Finished... Best Model Saved at {best_epoch} with Val Accuracy {best_acc:.4f}")



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def predict(self, x):
        logits = self.forward(x)
        preds = logits.argmax(-1)
        return preds, logits


def get_dataloader(dataset):

    train_loader = DataLoader(dataset["train"],
                              batch_size=100,
                              shuffle=True,
                              num_workers=1)

    val_loader = DataLoader(dataset["val"],
                              batch_size=10000,
                              shuffle=True,
                              num_workers=1)

    test_loader = DataLoader(dataset["test"],
                             batch_size=10000,
                             shuffle=True,
                             num_workers=1)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_dataset():

    train_val_data = datasets.MNIST(
        root="~/../../data/junhyun/",
        train=True,
        transform=ToTensor(),
        download=True
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=0)
    train_idx, val_idx =  list(splitter.split(train_val_data.data, train_val_data.targets))[0]

    train_data = copy(train_val_data)
    train_data.data = train_val_data.data[train_idx]
    train_data.targets = train_val_data.targets[train_idx]

    val_data = copy(train_val_data)
    val_data.data = train_val_data.data[val_idx]
    val_data.targets = train_val_data.targets[val_idx]
    val_data.train = False

    test_data = datasets.MNIST(
        root="~/../../data/junhyun/",
        train=False,
        transform=ToTensor(),
    )

    return {"train": train_data, "val": val_data, "test": test_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "-e", type=str, default="debug")
    parser.add_argument("--seed", "-s", type=int, default=0)
    args = parser.parse_args()

    main(args)
