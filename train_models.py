# This file was used to generate a couple of Base_RNN's with different hidden size params
# not important for the notebook/outcome, but I thought I'd include it anyway

from models import Base_RNN, GRU_RNN, LSTM_RNN
import torch
import torch.nn as nn
from pipeline import train, evaluate, AfterEpochCallbackParams
from sklearn.model_selection import train_test_split
from data_generation import generate_longterm_data
import os

NUM_SEQ = 1000
HORIZON=20
data_set, labels = generate_longterm_data(NUM_SEQ, variable_steps=False, noise=True, horizon=HORIZON)

X_train, X_test, y_train, y_test = train_test_split(data_set, labels, test_size=0.2, random_state=42)

LOG_FILE_PATH = "./logs/history"
MODEL_DESCRIPTION = ""

def write_stats_callback(params: AfterEpochCallbackParams):
    model_name = params.model.__class__.__name__
    filepath = f"{LOG_FILE_PATH}/{model_name}-{MODEL_DESCRIPTION}"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open(f"{filepath}/training-stats.csv", "a+") as f:
        header, content = params.as_csv_string()
        if params.epoch == 1:
            f.write(header + "\n")
        f.write(content  + "\n")

hidden_steps = [5,10,20,30,50]
EPOCHS = 250
mseloss = torch.nn.MSELoss()

for hidden_len in hidden_steps:
    print("Starting to train Base Model with Hidden Len: ", hidden_len)
    MODEL_DESCRIPTION = f"h-{hidden_len}"
    model = Base_RNN(1, hidden_len, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train(model, EPOCHS, X_train, y_train, mseloss, optimizer, post_epoch_callbacks=[write_stats_callback])
    torch.save(model, f"finished-base-{hidden_len}.pt")