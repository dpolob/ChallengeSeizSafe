"""
Clase de la red neuronal LSTM
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=1, dropout=0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Definir capa LSTM
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)
        # Definir capa de salida: Linear + Dropout layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.drop = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax()

    def init_hidden(self):
        # Inicializador de los estados ocultos
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, entrada):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(entrada)
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.drop(y_pred)
        y_pred = self.logsoftmax(y_pred)
        return y_pred
