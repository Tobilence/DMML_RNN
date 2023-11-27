from data_generation import generate_longterm_data
import numpy as np
import torch.nn as nn
import torch
from typing import List, Callable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class TrainingStats:
    train_loss: list[float]
    val_loss: list[float]

    def display_training_stats(self):
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.legend()

@dataclass
class AfterEpochCallbackParams:
    model: nn.Module
    epoch: int
    current_train_loss: float
    current_val_loss: float
    val_loss_history: float # the entire history, including the current val loss

    def as_csv_string(self, seperation_char=";") -> (str, str):
        """ Writes the parameter fields into a csv style string.
        Returns: 
            - the header for the CSV
            - the content of the class as CSV 
        """
        csv_content = f"{self.epoch}{seperation_char}{self.current_train_loss}{seperation_char}{self.current_val_loss}"
        return f"Epoch{seperation_char}Training_Loss{seperation_char}Validation_Loss", csv_content


def train(
    rnn: nn.Module, 
    epochs: int, 
    training_data: torch.FloatTensor, 
    training_labels: torch.FloatTensor, 
    loss_f: nn.Module,
    optimizer: torch.optim.Optimizer,
    post_epoch_callbacks: list[Callable] = [],
    validation_split_size = 0.10,
    random_seed=42,
    plot_loss=True,
    horizon=20,
    device=torch.device("cpu")
    ) -> TrainingStats:
    
    training_epoch_losses = []
    val_epoch_losses = []

    X_train, X_val, y_train, y_val = train_test_split(
        training_data, 
        training_labels, 
        test_size=validation_split_size,
        shuffle=True, 
        random_state=random_seed
    )

    for epoch in range(1, epochs+1):
        training_epoch_loss = []

        # iterate over training data
        for data_series, label_series in zip(X_train, y_train):
            label_series = torch.tensor(label_series).to(device)
            rnn.zero_grad()
            outputs = rnn.forward(data_series, horizon=horizon, device=device)
            loss = loss_f(outputs, label_series) 
            loss.backward()
            optimizer.step()
            training_epoch_loss.append(loss.cpu().detach().numpy())
        
        # test model on validation set
        with torch.no_grad():
            rnn.zero_grad()
            y_hat_vals = []
            val_losses = []
            for val_x, val_y in zip(X_val, y_val):
                y_hat_val = rnn.forward(val_x, horizon=horizon, device=device)
                val_loss = loss_f(y_hat_val, torch.tensor(val_y).to(device))
                y_hat_vals.append(y_hat_val)
                val_losses.append(val_loss.cpu().detach().numpy())
        
        # save epoch stats
        val_epoch_losses.append(np.mean(val_losses))
        training_epoch_losses.append(np.mean(training_epoch_loss))

        for callback in post_epoch_callbacks:
            callback(AfterEpochCallbackParams(rnn, epoch, training_epoch_losses[-1], val_epoch_losses[-1], val_epoch_losses))

        print(f"Finished epoch {epoch} of {epochs}. Avg Training Loss: {training_epoch_losses[-1]:2f} ; Avg Val Loss: {val_epoch_losses[-1]:2f}", end="\r")

    if plot_loss:
        plot_losses(TrainingStats(training_epoch_losses, val_epoch_losses))
    return TrainingStats(training_epoch_losses, val_epoch_losses)


def plot_losses(train_stats: TrainingStats):
    plt.plot(train_stats.train_loss, color="blue", label="Training Loss")
    plt.plot(train_stats.val_loss, color="orange", label="Validation Loss")
    plt.legend()

def evaluate(
        model: nn.Module,
        X_test,
        labels,
        loss_fn=nn.MSELoss(),
        visualize_samples=True,
        visualize_start_idx=0,
        visualize_oneline=False # if visualize_samples is true, decides whether 3 or 9 images are shown (and uses 3)
    ) -> (float, float):
    """
    Evaluates the model on the given test set.
    Returns a tuple of loss statistics: (mean_loss, median_loss)
    """
    with torch.no_grad():
        y_hats = []
        losses = []
        for x, y in zip(X_test, labels):
            y_hats.append(model(x))
            loss = loss_fn(y_hats[-1], torch.tensor(y))
            losses.append(loss.detach().numpy())
        
        if visualize_samples:
            if visualize_oneline:
                _, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))
                for i, (data, preds, label) in enumerate(zip(
                    X_test[visualize_start_idx:visualize_start_idx+4], 
                    y_hats[visualize_start_idx:visualize_start_idx+4], 
                    labels[visualize_start_idx:visualize_start_idx+4])
                ):
                    axs[i].plot(data, "o")
                    axs[i].plot(range(len(data), len(data)+len(label)), label, "o", label="Labels", color="green")
                    axs[i].plot(range(len(data), len(data)+len(label)), preds, "o", label="Predictions", color="darkorange")
            else:
                _, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
                for i, (data, preds, label) in enumerate(zip(
                    X_test[visualize_start_idx:visualize_start_idx+9], 
                    y_hats[visualize_start_idx:visualize_start_idx+9], 
                    labels[visualize_start_idx:visualize_start_idx+9])
                ):
                    row_idx = i % 3
                    col_idx = int(i / 3)
                    axs[col_idx][row_idx].plot(data, "o")
                    axs[col_idx][row_idx].plot(range(len(data), len(data)+len(label)), label, "o", label="Labels", color="green")
                    axs[col_idx][row_idx].plot(range(len(data), len(data)+len(label)), preds, "o", label="Predictions", color="darkorange")
    return np.mean(losses), np.median(losses)

def compare_models(models: list[nn.Module], X_test, y_test):
    losses = [evaluate(model, X_test, y_test, visualize_samples=False)[0] for model in models]
    models = [model.name for model in models]

    plt.bar(models, losses)

    plt.xlabel('Models')
    plt.ylabel('Loss')
    plt.title('Comparison of Model Losses')
    plt.xticks(rotation=90)

    plt.show()
