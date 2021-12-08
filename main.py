import os
import argparse
import logging
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import StratifiedShuffleSplit

from model import INPUT_SIZE, CustomCNN, initialize_model


NUM_CLASSES = {
        "mnist": 10,
        "emnist": 47,
        "fmnist": 10,
}

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

    os.makedirs(f"{args.mode}/{args.dataset}/", exist_ok=True)
    file_handler = logging.FileHandler(f"{args.mode}/{args.dataset}/{args.exp}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    num_classes = NUM_CLASSES[args.dataset]
    input_size = INPUT_SIZE[args.cnn]
    dataset = get_dataset(args.dataset, input_size)

    if args.mode == "train":
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
        plt.savefig(f"{args.mode}/{args.dataset}/examples.png")
        plt.close()
        logger.info("")
        logger.info("Val Dataset Config")
        logger.info("-" * 30)
        logger.info(dataset["val"])
        logger.info(dataset["val"].data.size())
        logger.info("")
    else:
        logger.info("Test Dataset Config")
        logger.info("-" * 30)
        logger.info(dataset["test"])
        logger.info(dataset["test"].data.size())
        logger.info("")

    dataloader = get_dataloader(dataset)

    model = CNN(args.cnn, num_classes)
    logger.info("Model Config")
    logger.info("-" * 30)
    logger.info(model)
    logger.info("")

    loss_func = nn.CrossEntropyLoss()
    logger.info("Loss Function Config")
    logger.info("-" * 30)
    logger.info(loss_func)
    logger.info("")

    rgb = (args.cnn != "custom")

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        logger.info("Optimizer Config")
        logger.info("-" * 30)
        logger.info(optimizer)
        logger.info("")

        train(dataloader, model, loss_func, optimizer, rgb=rgb)

    elif args.mode == "test":
        model_dict = torch.load(f"/data/junhyun/ckpt/{args.dataset}/{args.exp}.tar", map_location="cpu")
        model.load_state_dict(model_dict)
        model.eval()
        loss = 0
        acc = 0
        num_batch_test = len(dataloader["test"])
        for x, y in dataloader["test"]:
            if rgb:
                x = x.repeat(1, 3, 1, 1)
            with torch.no_grad():
                preds, logits = model.predict(x)
                loss_batch = loss_func(logits, y)
                loss += loss_batch
                acc_batch = (preds.cpu() == y).sum() / len(y)
                acc += acc_batch
        loss /= num_batch_test
        acc /= num_batch_test
        logger.info(f"Test_Loss: {loss.item():.4f}, Test_Acc: {acc.item():.4f}")

    elif args.mode == "ensemble":
        model_list = []
        with open(f"ensemble/{args.file}", "r") as f:
            exp_list = yaml.safe_load(f)
        for exp in exp_list:
            model = CNN(args.cnn, num_classes).to("cuda")
            model_dict = torch.load(f"/data/junhyun/ckpt/{args.dataset}/{exp}.tar")
            model.load_state_dict(model_dict)
            model.eval()
            model_list.append(model)

        acc = 0
        num_batch_test = len(dataloader["test"])
        for x, y in dataloader["test"]:
            if rgb:
                x = x.repeat(1, 3, 1, 1)
            probs = torch.zeros((len(dataset["test"]), num_classes))
            for model in model_list:
                with torch.no_grad():
                    _, logits = model.predict(x.to("cuda"))
                    probs += nn.Softmax(dim=-1)(logits.cpu())
            probs /= len(model_list)
            preds = probs.argmax(-1)
            acc_batch = (preds.cpu() == y).sum() / len(y)
            acc += acc_batch
        acc /= num_batch_test
        logger.info(f"Ensemble Size: {len(model_list)}, Ensemble_Acc: {acc.item():.4f}")

    elif args.mode == "wensemble":
        model_list = []
        weight_list = []
        with open(f"wensemble/{args.file}", "r") as f:
            exp_list = yaml.safe_load(f)
        for exp in exp_list:
            model = CNN(args.cnn, num_classes).to("cuda")
            model_dict = torch.load(f"/data/junhyun/ckpt/{args.dataset}/{exp}.tar")
            model.load_state_dict(model_dict)
            model.eval()
            model_list.append(model)
            weight = torch.zeros(num_classes)
            for x, y in dataloader["val"]:
                if rgb:
                    x = x.repeat(1, 3, 1, 1)
                preds, logits = model.predict(x.to("cuda"))
                for c in range(num_classes):
                    idx = (preds==c).nonzero(as_tuple=True)[0]
                    num_correct = (y[idx] == c).sum().item()
                    acc = num_correct / len(idx)
                    weight[c] += acc
            weight /= weight.sum()
            weight_list.append(weight)

        acc = 0
        num_batch_test = len(dataloader["test"])
        for x, y in dataloader["test"]:
            if rgb:
                x = x.repeat(1, 3, 1, 1)
            probs = torch.zeros((len(dataset["test"]), num_classes))
            for model, weight in list(zip(model_list, weight_list)):
                with torch.no_grad():
                    preds, logits = model.predict(x.to("cuda"))
                    probs += nn.Softmax(dim=-1)(logits.cpu()) * weight[preds][:, None]
            probs /= probs.sum(-1)[:, None]
            preds = probs.argmax(-1)
            acc_batch = (preds.cpu() == y).sum() / len(y)
            acc += acc_batch
        acc /= num_batch_test
        logger.info(f"Ensemble Size: {len(model_list)}, Ensemble_Acc: {acc.item():.4f}")


def train(dataloader, model, loss_func, optimizer, num_epoch=20, print_epoch=1, eval_epoch=1, rgb=False):

    logger.info("Train Started ...")
    logger.info("-" * 30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batch_train = len(dataloader["train"])
    num_batch_val = len(dataloader["val"])

    best_acc = .0
    best_epoch = 0
    for epoch in range(num_epoch):
        model = model.to(device)
        model.train()
        loss = 0
        for batch_idx, (x, y) in enumerate(tqdm(dataloader["train"])):
            if rgb:
                x = x.repeat(1, 3, 1, 1)
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
                if rgb:
                    x = x.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    model = model.cpu()
                    preds, logits = model.predict(x)
                    loss_batch = loss_func(logits, y)
                    loss += loss_batch
                    acc_batch = (preds==y).sum() / len(y)
                    acc += acc_batch
            loss /= num_batch_val
            acc /= num_batch_val

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                logger.info(f"Val_Loss: {loss.item():.4f}, Val_Acc: {acc.item():.4f} ------> Best Model Saved")
                os.makedirs(f"/data/junhyun/ckpt/{args.dataset}/", exist_ok=True)
                torch.save(model.state_dict(), f"/data/junhyun/ckpt/{args.dataset}/{args.exp}.tar")
            else:
                logger.info(f"Val_Loss: {loss.item():.4f}, Val_Acc: {acc.item():.4f}")

    logger.info(f"")
    logger.info(f"Train Finished... Best Model Saved at {best_epoch} with Val Accuracy {best_acc:.4f}")


class CNN(nn.Module):

    def __init__(self, model_type, num_classes):
        super(CNN, self).__init__()
        if model_type == "custom":
            model = CustomCNN(num_classes)
        else:
            model, input_size = initialize_model(model_type, num_classes, feature_extract=True, use_pretrained=True)
        self.cnn = model

    def forward(self, x):
        return self.cnn(x)

    def predict(self, x):
        logits = self(x)
        preds = logits.argmax(-1)
        return preds, logits


def get_dataloader(dataset):

    train_loader = DataLoader(dataset["train"],
                              batch_size=50,
                              shuffle=True,
                              num_workers=1)

    val_loader = DataLoader(dataset["val"],
                            batch_size=dataset["val"].data.shape[0],
                            shuffle=True,
                            num_workers=1)

    test_loader = DataLoader(dataset["test"],
                             batch_size=dataset["test"].data.shape[0],
                             shuffle=True,
                             num_workers=1)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_dataset(dataset, input_size):

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ]),
        # 'val': transforms.Compose([
        #     # transforms.Resize(input_size),
        #     # transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ]),
    }
    if dataset == "mnist":
        train_val_data = datasets.MNIST(
            root="/data/junhyun/",
            train=True,
            # transform=ToTensor(),
            download=True,
            transform=data_transforms["train"]
        )
        test_data = datasets.MNIST(
            root="/data/junhyun/",
            train=False,
            # transform=ToTensor(),
            transform=data_transforms["train"]
        )

    elif dataset == "emnist":
        train_val_data = datasets.EMNIST(
            split="balanced",
            root="/data/junhyun/",
            train=True,
            # transform=ToTensor(),
            download=True,
            transform=data_transforms["train"]
        )
        test_data = datasets.EMNIST(
            split="balanced",
            root="/data/junhyun/",
            train=False,
            # transform=ToTensor(),
            transform=data_transforms["train"]
        )

    elif dataset == "fmnist":
        train_val_data = datasets.FashionMNIST(
            root="/data/junhyun/",
            train=True,
            # transform=ToTensor(),
            download=True,
            transform=data_transforms["train"]
        )
        test_data = datasets.FashionMNIST(
            root="/data/junhyun/",
            train=False,
            # transform=ToTensor(),
            transform=data_transforms["train"]
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

    return {"train": train_data, "val": val_data, "test": test_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "-e", type=str, default="debug")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--mode", "-m", choices=["train", "test", "ensemble", "wensemble"], default="train")
    parser.add_argument("--file", "-f", type=str, default=None)
    parser.add_argument("--dataset", "-d", choices=["mnist", "emnist", "fmnist"], default="mnist")
    parser.add_argument("--cnn", "-c", choices=["custom", "resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"], default="custom")
    args = parser.parse_args()
    if args.mode == "ensemble":
        assert args.file is not None, ValueError("File path required for ensemble mode.")

    main(args)
