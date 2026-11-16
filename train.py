"""Train the CNN on the training set, watch validation loss, save the weights to model.pth.

Never touches the test set. evaluate.py owns that.
"""
import torch
from torch import nn


def train(model, train_loader, val_loader, epochs=10, lr=0.001, save_path="model.pth", seed=0):
    torch.manual_seed(seed)  # same seed = same result every run (run seeds 0, 1, 2 to check it's not luck)

    loss_fn = nn.CrossEntropyLoss()  # scores how wrong the 2 output scores (DOWN, UP) are
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # nudges the weights to lower the loss

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):  # 1 epoch = the model sees every training picture once
        model.train()  # training mode
        total = 0.0
        for images, labels in train_loader:  # one small batch of pictures at a time
            optimizer.zero_grad()  # clear last batch's nudges
            loss = loss_fn(model(images), labels)  # guess, then score the guess
            loss.backward()  # work out which way to nudge each weight
            optimizer.step()  # nudge them
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()  # test mode: no learning
        with torch.no_grad():  # skip the nudge math, faster
            val_loss = sum(loss_fn(model(x), y).item() for x, y in val_loader) / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")

    torch.save(model.state_dict(), save_path)  # state_dict = just the learned numbers
    return history


if __name__ == "__main__":
    # These two files are not done yet (Nov 16). Names agreed with Arnav and Dale.
    from model import StockCNN
    from data_preprocessing import get_dataloaders

    train_loader, val_loader, _ = get_dataloaders()  # _ = test loader, unused on purpose
    # Hyperparameters (write the ones you try here): lr=0.001, epochs=10, seed=0
    train(StockCNN(), train_loader, val_loader, epochs=10, lr=0.001)
