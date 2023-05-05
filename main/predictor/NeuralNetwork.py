import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from functools import partial
import joblib


class Architecture(nn.Module):
    def __init__(self,X):
        super(Architecture, self).__init__()
        L = 128
        P = 2*L
        M = 2*P
        input_shape = X.shape[1]
        self.fc1 = nn.Linear(input_shape, M)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(M,P)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(P,L)
        self.fc4 = nn.Linear(L,1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        # x = F.sigmoid(x)
        return x


class NeuralNetwork():
    def __init__(self,X):
        torch.manual_seed(314)
        self.model = Architecture(X)
        self.epoch = 5
        self.batch_size = 10
        self.criterion = F.mse_loss
    
    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame):    
        X = torch.tensor(x_train.values, dtype=torch.float32)
        Y = torch.tensor(y_train.values, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        
        for _ in tqdm(range(self.epoch)):
    
            for batch in train_loader:
                # sampling
                x, t = batch
                
                # reset param's gradient
                optimizer.zero_grad()
                
                y = self.model(x)[:,0]
                
                loss = self.criterion(y, t)
                
                # calc gradient
                loss.backward()
            
                # update
                optimizer.step()
    
    def predict(self, x_test:pd.DataFrame):
        self.model.eval()
        index = x_test.index
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_predict = self.model(x_test)
        y_predict = y_predict.detach().numpy()
        y_predict = pd.DataFrame(y_predict,index=index,columns=['prediction'])
        return y_predict
        
