import torch
SEED = 42
DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print("Device Selected:", DEVICE)
    return DEVICE


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed)



# import torch
# from torch import nn, optim
# from torch.nn import functional as F
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from collections import defaultdict


# from Dataset import CIFAR10

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# def correct_pred_count(prediction, labels):
#     return prediction.argmax(dim=1).eq(labels).sum().item()

# def incorrect_pred_count(prediction, labels):
#     prediction = prediction.argmax(dim = 1)
#     indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
#     return indices, prediction[indices].tolist(), labels[indices].tolist()

# class Train(object):
#     def __init__(self, model, dataset, criterion, optimizer, l1= 0):
#         self.model = model
#         self.dataset = dataset
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.l1 = l1

#         self.train_losses = []
#         self.train_acc = []

#     def __call__(self):
#         self.model.train()
#         pbar = tqdm(self.dataset.train_loader)

#         train_loss = 0
#         correct = 0
#         processed = 0

#         for batch_idx, (data, target) in enumerate(pbar):
#             data, target = data.to(self.device), target.to(self.device)
#             self.optimizer.zero_grad()

#             pred = self.model(data)

#             loss = self.criterion(pred,target)
#             if self.l1 > 0:
#                 loss += self.l1 * sum(p.abs().sum() for p in self.model.parameters())

#             train_loss += loss.item() * len(data)

#             loss.backward()
#             self.optimizer.step()


#             correct += correct_pred_count(pred, target)
#             processed += len(data)

#             pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Train_Accuracy={100 * correct / processed:0.2f}')

#         self.train_acc.append(100* correct/ processed)
#         self.train_losses.append(train_loss/ processed)

#     def plot_stats(self):
#         fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#         axs[0].plot(self.train_losses)
#         axs[0].set_title("Training Loss")
#         axs[1].plot(self.train_acc)
#         axs[1].set_title("Training Accuracy")

# class Test(object):
#     def __init__(self, model, dataset, criterion):
#         self.model = model
#         self.device = device
#         self.criterion = criterion
#         self.dataset = dataset

#         self.test_losses = []
#         self.test_acc = []

#     def __call__(self, incorrect_preds= None):
#         self.model.eval()

#         test_loss = 0
#         correct = 0
#         processed = 0

#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(self.dataset.test_loader):
#                 data,target = data.to(self.device), target.to(self.device)
#                 pred = self.model(data)

#                 test_loss += self.criterion(pred, target, reduction="sum").item()

#                 correct += correct_pred_count(pred, target)
#                 processed += len(data)

#                 if incorrect_preds is not None:
#                     ind, pred, truth = incorrect_pred_count(pred, target)
#                     incorrect_preds["images"] += data[ind]
#                     incorrect_preds["ground_truths"] += truth
#                     incorrect_preds["predicted_vals"] += pred


#         test_loss /= processed
#         self.test_acc.append(100 * correct / processed)
#         self.test_losses.append(test_loss)

#         print(f"Test: Average loss: {test_loss:0.4f}, Test_Accuracy: {100 * correct / processed:0.2f}")

#         return test_loss

#     def plot_stats(self):
#         fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#         axs[0].plot(self.test_losses)
#         axs[0].set_title("Test Loss")
#         axs[1].plot(self.test_acc)
#         axs[1].set_title("Test Accuracy")

# # class Experiment():
# #     def __init__(self, model, dataset, lr=0.03, criterion = None, epochs = 24):
# #         self.device = device
# #         self.model = model.to(device)
# #         self.dataset = dataset
# #         self.criterion = criterion or nn.CrossEntropyLoss()
# #         self.best_lr = self.find_best_LR()
# #         self.epochs = epochs
# #         self.optimizer = optim.Adam(self.model.parameters(), lr=lr, momentum = 0.9)
# #         self.scheduler = optim.lr_scheduler.OneCycleLR(
# #             self.optimizer,
# #             max_lr=self.best_lr,
# #             steps_per_epoch=len(self.dataset.train_loader),
# #             epochs=self.epochs,
# #             pct_start=5/self.epochs,
# #             div_factor=100,
# #             three_phase=False,
# #             final_div_factor=100,
# #             anneal_strategy='linear'
# #         )

# #         self.train = Train(self.model, dataset, self.criterion, self.optimizer, self.scheduler)
# #         self.test = Test(self.model, dataset, self.criterion)
# #         self.incorrect_preds = None
        

# #     def find_best_LR(self):
# #         lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device= self.device)
# #         lr_finder.range_test(self.dataset.train_loader, end_lr= 0.1, num_iter= 100, step_mode='exp')
# #         _, best_lr = lr_finder.plot()
# #         lr_finder.reset()
# #         return best_lr
    
# #     def execute(self, target=None):
# #         target_count = 0
# #         for epoch in range(1, self.epochs + 1):
# #             print(f'Epoch: {epoch}')
# #             self.train()
# #             test_loss, test_acc = self.test()
# #             if target is not None and test_acc >= target:
# #                 target_count += 1
# #                 if target_count >= 3:
# #                     print("Target Validation accuracy achieved thrice. Stopping Training.")
# #                     break
    
# # class Experiment(object):
# #     def __init__(self, model, dataset, criterion=None, epochs=24):
# #         self.device = device
# #         self.model = model.to(self.device)
# #         self.dataset = dataset
# #         self.criterion = criterion or nn.CrossEntropyLoss()
# #         self.epochs = epochs
# #         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=1e-2)
# #         self.best_lr = self.find_lr()
# #         self.scheduler = optim.lr_scheduler.OneCycleLR(
# #             self.optimizer,
# #             max_lr=self.best_lr,
# #             steps_per_epoch=len(self.dataset.train_loader),
# #             epochs=self.epochs,
# #             pct_start=5/self.epochs,
# #             div_factor=100,
# #             three_phase=False,
# #             final_div_factor=100,
# #             anneal_strategy='linear'
# #         )
# #         self.train = Train(self.model, dataset, self.criterion, self.optimizer, self.scheduler)
# #         self.test = Test(self.model, dataset, self.criterion)
# #         self.incorrect_preds = None

# #     def find_lr(self):
# #         lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
# #         lr_finder.range_test(self.dataset.train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
# #         _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
# #         lr_finder.reset()  # to reset the model and optimizer to their initial state
# #         return best_lr

# #     def execute(self, target=None):
# #         target_count = 0
# #         for epoch in range(1, self.epochs + 1):
# #             print(f'Epoch: {epoch}')
# #             self.train()
# #             test_loss, test_acc = self.test()
# #             if target is not None and test_acc >= target:
# #                 target_count += 1
# #                 if target_count >= 3:
# #                     print("Target Validation accuracy achieved thrice. Stopping Training.")
# #                     break

    
        



