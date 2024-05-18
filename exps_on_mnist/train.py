import torch
from torch import nn

import matplotlib.pyplot as plt

from IPython.display import clear_output


def train_one_epoch(model, loss_fn, optimizer, training_loader, device=torch.device("cpu")):
    #running_loss = 0.
    #last_loss = 0.
    model.to(device)
    model.train()
    losses = []

    for i, data in enumerate(training_loader):
        inputs, labels = data
        # print(labels)
        optimizer.zero_grad()
        logits = model(inputs.to(device), device)
        loss = loss_fn(logits, labels.to(device))
        # print("Logits:", logits, "Labels:", labels)
        # print("Loss:", loss, nn.CrossEntropyLoss()(logits, labels))
        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)
        optimizer.step()
        #running_loss += loss.item()
        losses.append(loss.item())

    return losses

def train(model, loss_fn, optimizer, training_loader, epochs=100, device=torch.device("cpu")):
    losses = []
    for _ in range(epochs):
        losses += train_one_epoch(model, loss_fn, optimizer, training_loader, device)
        plt.plot(losses)
        plt.show()
        clear_output(wait=True)
    

def evaluate_acc(model, data_loader, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    correct_preds = 0
    
    for i, data in enumerate(data_loader):
        inputs, labels = data
        labels_pred = model.predict(inputs.to(device), device=device)
        correct_preds += (labels_pred == labels.to(device)).sum()
    print(correct_preds, len(data_loader.dataset))
    print(correct_preds / len(data_loader.dataset))