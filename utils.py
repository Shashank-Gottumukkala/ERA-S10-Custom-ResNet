import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def correct_pred_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()

def incorrect_pred_count(prediction, labels):
    prediction = prediction.argmax(dim = 1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

class Train(object):
    def __init__(self, model, dataset, criterion, optimizer, l1= 0):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.l1 = l1

        self.train_losses = []
        self.train_acc = []

    def __call__(self):
        self.model.train()
        pbar = tqdm(self.dataset.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(data)

            loss = self.criterion(pred,target)
            if self.l1 > 0:
                loss += self.l1 * sum(p.abs().sum() for p in self.model.parameters())

            train_loss += loss.item() * len(data)

            loss.backward()
            self.optimizer.step()


            correct += correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Train_Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100* correct/ processed)
        self.train_losses.append(train_loss/ processed)

    def plot_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(self.train_losses)
        axs[0].set_title("Training Loss")
        axs[1].plot(self.train_acc)
        axs[1].set_title("Training Accuracy")

