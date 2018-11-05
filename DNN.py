"""
Deep Neural NETWORK
Created: Diego

"""
import torch
import torch.nn as nn
import torch.utils.data


class DeepNeuralNetwork(nn.Module):
    def __init__(self, inputLayer, hidden1Layer, hidden2Layer, outputLayer):
        super(DeepNeuralNetwork, self).__init__()
        #Configuracion de la red neuronal
        self.fc1 = nn.Linear(inputLayer, hidden1layer)
        self.fc2 = nn.Linear(hidden1Layer, hidden2Layer)
        self.fc3 = nn.Linear(hidden2Layer, outputLayer)
        #Funciones de activacion
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.logsm1 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.logsm1(out)
        return(out)

def EntrenarRed(datos, dnn, **kwargs):
    learningRate = kwargs['learningRate']
    numEpoch = kwargs['numEpoch']
    batchSize = kwargs['batchSize']
    
    lossFN = nn.NLLoss()
    optimizer = torch.optim.Adam(dnn.parameters(), lr=learningRate)

    #Preparacion del dataset con dataloader
    trainLoader = torch.utils.data.DataLoader(dataset=torch.tensor(datos, dtype=torch.float), 
            batch_size=batchSize, shuffle=True)

    for epoch in range(numEpoch):
        for i, data in enumerate(trainLoader, start=1):
            salidas = torch.tensor(data[:,-1], dtype=torch.float, requires_grad=True)
            entrada = torch.tensor(data[:,:-1], dtype=torch.float, requires_grad=True)
            optimizer.zero_grad()
            outputs = dnn(entrada)
            loss = lossFN(outputs, salidas)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {i}/{numEpoch} - Loss: {loss.item()}")






