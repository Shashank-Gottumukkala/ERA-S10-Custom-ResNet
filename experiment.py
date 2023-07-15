from torch import nn, optim
from matplotlib import pyplot as plt
from collections import defaultdict
from torch_lr_finder import LRFinder

from utils import get_device
from backprop import Train, Test


class Experiment(object):
    def __init__(self, model, dataset, criterion=None, epochs=24):
        self.device = get_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.03, weight_decay=1e-2)
        self.best_lr = self.find_lr()
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.best_lr,
            steps_per_epoch=len(self.dataset.train_loader),
            epochs=self.epochs,
            pct_start=5/self.epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        self.train = Train(self.model, dataset, self.criterion, self.optimizer, self.scheduler)
        self.test = Test(self.model, dataset, self.criterion)
        self.incorrect_preds = None

    def find_lr(self):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
        lr_finder.range_test(self.dataset.train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot() 
        lr_finder.reset()  
        return best_lr

    def execute(self, target=None):
        target_count = 0
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}')
            self.train()
            test_loss, test_acc = self.test()
            if target is not None and test_acc >= target:
                target_count += 1
                if target_count >= 3:
                    print("Target Validation accuracy achieved thrice. Stopping Training.")
                    break