import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import datetime

import torch.optim as optim


def create_sequences(data: pd.DataFrame, labels: pd.DataFrame, seq_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if seq_type == "classification":
        targets_tensor = torch.tensor(np.array(targets).flatten(), dtype=torch.long) - 1
    else:
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
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
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
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

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))  # (input, hidden, and cell state)
        hn = hn[-1]  # last layer's hidden state
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)
        return out


def train_model(n_epochs, model, optimiser, loss_func, train_loader, test_loader, save_path='result/train_loss.txt'):
    model.train()
    for epoch in range(n_epochs):
        # Training mode
        for inputs, label in train_loader:
            # 把数据移动到 CUDA 上
            inputs = inputs.to(device)
            label = label.to(device)
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
            with open(save_path, 'a') as f:
                f.write(f"Epoch: {epoch}, train loss: {loss.item():.5f}\n")
            # print("Result std: "+str(np.std(np.array((10-1)*y_test+1))))
            # print("Prediction std: "+str(np.std(np.array((10-1)*test_preds+1))))
    return model


def evaluate_model_classification(model, test_loader, loss_func, save_path) -> List:
    prediction = []
    y_labels = []
    model.eval()
    # Disable gradient calc
    with torch.no_grad():
        test_loss = 0.0
        # Compute classes and losses
        for inputs, label in test_loader:
            # 把数据移动到 CUDA 上
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            test_loss += loss.item() * inputs.size(0)
            prediction.append(outputs.cpu())
            y_labels.append(label.cpu())
    
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

    result_str = classification_report(y_labels, pred_labels)
    # mae = mean_absolute_error(y_labels, pred_labels)
    # mse = mean_squared_error(y_labels, pred_labels)
    # rmse = math.sqrt(mse)
    # # 格式化结果
    # result_str = (
    #     f"Test Loss : {test_loss:.6f}\n"
    #     f"MSE       : {mse:.6f}\n"
    #     f"MAE       : {mae:.6f}\n"
    #     f"RMSE      : {rmse:.6f}\n"
    #     f"Accuracy  : {accuracy:.6f}\n"
    #
    # )
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_str += f"Time      : {now}\n"

    # 保存到本地文件
    with open(save_path, 'a') as f:
        f.write(result_str)
    

    # 保存预测结果和真实标签到 CSV
    df_classification_result = pd.DataFrame({
        'True': y_labels,
        'Predicted': pred_labels
    })
    df_classification_result.to_csv('result/pred_vs_true_classification.csv', index=False)

    return [test_loss, pred_labels, y_labels]


def evaluate_model_regression(model, test_loader, loss_func, accuracy_threshold=0.1, save_path='result/evaluation_regression.txt',):
    prediction = []
    y_labels = []
    model.eval()
    # Disable gradient calc
    with torch.no_grad():
        test_loss = 0.0
        # Compute classes and losses
        for inputs, label in test_loader:
            # 把数据移动到 CUDA 上
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            test_loss += loss.item() * inputs.size(0)
            prediction.append(outputs)
            y_labels.append(label)
    test_loss /= len(test_loader.dataset)

    # 合并所有 batch 的预测和标签
    prediction = torch.cat(prediction).squeeze()
    y_labels = torch.cat(y_labels).squeeze()

    if prediction.dim() > 1 and prediction.size(1) > 1:
        prediction = prediction[:, 0]

    # 计算指标
    mse = F.mse_loss(prediction, y_labels).item()
    mae = F.l1_loss(prediction, y_labels).item()
    rmse = torch.sqrt(F.mse_loss(prediction, y_labels)).item()
    acc = (torch.abs(prediction - y_labels) < accuracy_threshold).float().mean().item()
    r_squared = r2_score(y_labels, prediction)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 格式化结果
    result_str = (
        f"Test Loss : {test_loss:.6f}\n"
        f"MSE       : {mse:.6f}\n"
        f"MAE       : {mae:.6f}\n"
        f"RMSE      : {rmse:.6f}\n"
        f"Accuracy  : {acc:.6f} (Threshold={accuracy_threshold})\n"
        f"R_Squared      : {r_squared:.6f}\n"
        
    )
    result_str += f"Time      : {now}\n"

    # 创建 DataFrame 保存预测值和真实值
    df_result = pd.DataFrame({
        'True': y_labels.cpu().numpy(),
        'Predicted': prediction.cpu().numpy()
    })

    # 可选：保存为 CSV 文件，方便后续画图
    df_result.to_csv('result/pred_vs_true_regression.csv', index=False)

    # 保存到本地文件
    with open(save_path, 'a') as f:
        f.write(result_str)



def training_pipeline(train_loader: DataLoader, test_loader: DataLoader, model_type: str):
    # Interval time: 1800, 7200, 21600, 36000
    # Learning rate 0.1, 0.05, 0.001, 0.0005
    # Hidden_size 2, 4, 10, 15, 25, 35
    # Num layers 1, 2, 4,8

    n_epochs = 100  # 1000 epochs
    learning_rates = [0.0005]

    input_size = 10  # number of features
    hidden_sizes = [25]  # number of features in hidden state [5, 15, 25]
    num_layers_list = [2]  # number of stacked lstm layers [1, 2, 4]



    num_classes = 5  # number of output classes

    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                print(
                    f"Testing learning rate: {learning_rate}, features in hidden layer {hidden_size}, stacked LSTM {num_layers}")

                if model_type == "regression":
                    save_path='result/evaluation_regression_n_epochs'+str(n_epochs)+'_hidden_sizes'+str(hidden_sizes)+'_num_layers_list'+str(num_layers_list)+'.txt'
                    save_path_loss='result/loss_regression_n_epochs'+str(n_epochs)+'_hidden_sizes'+str(hidden_sizes)+'_num_layers_list'+str(num_layers_list)+'.txt'
                    lstm_regression = LSTMRegression(num_classes, input_size, hidden_size, num_layers).to(device)
                    optimiser = torch.optim.Adam(lstm_regression.parameters(), lr=learning_rate)
                    loss_func = torch.nn.MSELoss()  # mean-squared error for regression
                    trained_model_regression = train_model(n_epochs=n_epochs, model=lstm_regression,
                                                           optimiser=optimiser,
                                                           loss_func=loss_func, train_loader=train_loader,
                                                           test_loader=test_loader,
                                                           save_path = save_path_loss)
                    evaluate_model_regression(trained_model_regression, test_loader, loss_func,save_path = save_path)
                else:
                    save_path='result/evaluation_classification_n_epochs'+str(n_epochs)+'_hidden_sizes'+str(hidden_sizes)+'_num_layers_list'+str(num_layers_list)+'.txt'
                    save_path_loss='result/loss_classification_n_epochs'+str(n_epochs)+'_hidden_sizes'+str(hidden_sizes)+'_num_layers_list'+str(num_layers_list)+'.txt'
                    lstm_classifier = LSTMClassifier(num_classes, input_size, hidden_size, num_layers).to(device)
                    optimiser = torch.optim.Adam(lstm_classifier.parameters(), lr=learning_rate)
                    loss_func = nn.CrossEntropyLoss()
                    trained_model_classification = train_model(n_epochs=n_epochs, model=lstm_classifier, optimiser=optimiser,
                                                                loss_func=loss_func, train_loader=train_loader,
                                                                test_loader=test_loader,
                                                                save_path = save_path_loss)
                    
                    [test_loss, pred_labels, y_labels] = evaluate_model_classification(trained_model_classification, test_loader, loss_func,save_path = save_path)
                    print(f"Test loss: {test_loss:.5f}")
                    equal_labels = pred_labels == y_labels
                    accuracy = np.sum(equal_labels) / len(equal_labels)
                    print(f"Accuracy: {accuracy:.5f}")

def run_model(task_type: str, df: pd.DataFrame):
    X = df.drop(columns=["Unnamed: 0", "id", "date", "mood_class", "average_mood"])

    if task_type == "regression":
        y = df[['group', 'average_mood']]

        x_seq, y_seq = create_sequences(X, y, "regression")
        train_size = int(0.75 * x_seq.size(0))
        test_size = x_seq.size(0) - train_size
        dataset = TensorDataset(x_seq, y_seq)

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        training_pipeline(train_loader, test_loader, "regression")
    else:
        y = df[['group', 'mood_class']]

        x_seq, y_seq = create_sequences(X, y, "classification")
        train_size = int(0.75 * x_seq.size(0))
        test_size = x_seq.size(0) - train_size
        dataset = TensorDataset(x_seq, y_seq)

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        training_pipeline(train_loader, test_loader, "classification")

if __name__ == '__main__':
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CSV_FILE = 'static/df_temporal.csv'
    df = pd.read_csv(CSV_FILE)
    # run_model(task_type="regression", df=df)
    run_model(task_type="classification", df=df)






