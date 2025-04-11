import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def create_classification_sequences(data, labels) -> Tuple[torch.Tensor, torch.Tensor]:
    time_window = 5
    sequences = []
    targets = []

    for group, sdf in list(data.groupby('group')):
        for i in range(len(sdf) - time_window):
            sdf_time_wind = sdf.drop(columns=['group']).iloc[i:i + time_window].to_numpy()
            sequences.append(sdf_time_wind)
    for group, sdf in list(labels.groupby('group')):
        for i in range(len(sdf) - time_window):
            sdf_time_wind = sdf.drop(columns=['group']).iloc[i + time_window].to_numpy()
            targets.append(sdf_time_wind)

    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets).flatten(), dtype=torch.long) - 1
    return sequences_tensor, targets_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, output_size)  # fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))  # (input, hidden, and cell state)
        hn = hn[-1]  # last layer's hidden state
        out = self.fc_1(hn)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)
        out = self.softmax(out)
        return out


class LSTMRegression(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, output_size)  # fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))  # (input, hidden, and cell state)
        hn = hn[-1]  # last layer's hidden state
        out = self.fc_1(hn)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)
        out = self.softmax(out)
        return out


def train_model(n_epochs, model, optimiser, loss_func, train_loader, test_loader):
    model.train()
    for epoch in range(n_epochs):
        # Training mode
        for inputs, label in train_loader:
            # Reset gradients
            optimiser.zero_grad()
            # Forward propagation
            outputs = model(inputs)
            # Training loss
            loss = loss_func(outputs, label)
            # print("loss.item(): ", loss.item())
            # print(loss)
            loss.backward()
            optimiser.step()
            # Update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimiser.step()

        # test_loss, pred_labels, y_labels = evaluate_model(model, test_loader, loss_func)
        if epoch % 10 == 0:
            # print(f"Epoch: {epoch}, train loss: {loss.item():.5f}, test loss: {test_loss:.5f}")
            print(f"Epoch: {epoch}, train loss: {loss.item():.5f}")
            # print("Result std: "+str(np.std(np.array((10-1)*y_test+1))))
            # print("Prediction std: "+str(np.std(np.array((10-1)*test_preds+1))))
    return model


def evaluate_model(model, test_loader, loss_func) -> List:
    prediction = []
    y_labels = []
    model.eval()
    # Disable gradient calc
    with torch.no_grad():
        test_loss = 0.0
        # Compute classes and losses
        for inputs, label in test_loader:
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            test_loss += loss.item() * inputs.size(0)
            prediction.append(outputs)
            y_labels.append(label)
    test_loss /= len(test_loader.dataset)
    prediction = np.array([np.array(tensor) for tensor in prediction])
    y_labels = np.array([np.array(tensor) for tensor in y_labels]).flatten()
    # print("prediction: ", prediction)

    pred_labels = np.array([np.argmax(current) + 1 for current in prediction])
    y_labels = np.array([current + 1 for current in y_labels])
    # print("y_labels2: ", y_labels)
    # print("pred_labels2: ", pred_labels)
    equal_labels = pred_labels == y_labels
    accuracy = np.sum(equal_labels) / len(equal_labels)
    print(f"Accuracy: {accuracy:.5f}")
    return [test_loss, pred_labels, y_labels]


def training_pipeline(train_loader: DataLoader, test_loader: DataLoader):
    # Interval time: 1800, 7200, 21600, 36000
    # Learning rate 0.1, 0.05, 0.001, 0.0005
    # Hidden_size 2, 4, 10, 15, 25, 35
    # Num layers 1, 2, 4,8

    n_epochs = 80  # 1000 epochs
    learning_rates = [0.001]

    input_size = 12  # number of features
    hidden_sizes = [5, 15, 25]  # number of features in hidden state [5, 15, 25]
    num_layers_list = [1, 2, 4]  # number of stacked lstm layers [1, 2, 4]

    num_classes = 5  # number of output classes

    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                print(
                    f"Testing learning rate: {learning_rate}, features in hidden layer {hidden_size}, stacked LSTM {num_layers}")
                lstm_classifier = LSTMClassifier(num_classes, input_size, hidden_size, num_layers)
                # loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
                loss_func = nn.CrossEntropyLoss()
                optimiser = torch.optim.Adam(lstm_classifier.parameters(), lr=learning_rate)

                trained_model = train_model(n_epochs=n_epochs, model=lstm_classifier, optimiser=optimiser,
                                loss_func=loss_func, train_loader=train_loader, test_loader=test_loader)
                [test_loss, pred_labels, y_labels] = evaluate_model(trained_model, test_loader, loss_func)
                print(f"Test loss: {test_loss:.5f}")
                equal_labels = pred_labels == y_labels
                accuracy = np.sum(equal_labels) / len(equal_labels)
                print(f"Accuracy: {accuracy:.5f}")

if __name__ == '__main__':
    CSV_FILE = 'static/df_temporal.csv'
    df = pd.read_csv(CSV_FILE)

    rnn = nn.LSTM(10, 20, 2)  # 一个单词向量长度为10，隐藏层节点数为20，LSTM有2层
    input = torch.randn(5, 3, 10)  # 输入数据由3个句子组成，每个句子由5个单词组成，单词向量长度为10
    h0 = torch.randn(2, 3, 20)  # 2：LSTM层数*方向 3：batch 20： 隐藏层节点数
    c0 = torch.randn(2, 3, 20)  # 同上
    output, (hn, cn) = rnn(input, (h0, c0))

    X = df.drop(columns=["Unnamed: 0", "id", "date", "mood_class", "average_mood"])
    # print(X.head())
    y = df[['group', 'mood_class']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
    # Create sequences for training and testing datasets
    train_sequences, train_targets = create_classification_sequences(X_train, y_train)
    test_sequences, test_targets = create_classification_sequences(X_test, y_test)
    # torch.Size([48, 39, 12]) torch.Size([48, 39]): 48 seqs, each seq has len 39, each cell in seq has 12 variables
    # print(train_sequences.size(), train_targets.size())
    #
    # print(test_targets)
    # print(train_targets)

    train_dataset = TensorDataset(train_sequences, train_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    training_pipeline(train_loader, test_loader)

