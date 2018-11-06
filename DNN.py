"""
Deep Neural NETWORK
Created: Diego

"""
import torch
import torch.nn as nn
import torch.utils.data


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_layer, hidden1_layer, hidden2_layer, output_layer):
        super(DeepNeuralNetwork, self).__init__()
        # Configuracion de la red neuronal
        self.fc1 = nn.Linear(input_layer, hidden1_layer)
        self.fc2 = nn.Linear(hidden1_layer, hidden2_layer)
        self.fc3 = nn.Linear(hidden2_layer, output_layer)
        # Funciones de activacion
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
        return out


def entrenarred(datos, dnn, learning_rate, num_epoch, batch_size):
    # learning_rate = kwargs['learningRate']
    # num_epoch = kwargs['numEpoch']
    # batch_size = kwargs['batchSize']

    lossFN = nn.NLLLoss()
    optimizer = torch.optim.Adam(dnn.parameters(), lr=learning_rate)

    # Preparacion del dataset con dataloader
    train_loader = torch.utils.data.DataLoader(dataset=torch.tensor(datos, dtype=torch.float),
                                              batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):
        for i, data in enumerate(train_loader, start=1):
            salidas = torch.tensor(data[:, -1], dtype=torch.float, requires_grad=True)
            entrada = torch.tensor(data[:, :-1], dtype=torch.float, requires_grad=True)
            optimizer.zero_grad()
            outputs = dnn(entrada)
            loss = lossFN(outputs, salidas.long())
            loss.backward()
            optimizer.step()

        print("Epoch {}/{} - Loss: {}".format(epoch, num_epoch, loss.item()))

def EvaluarRed(datos, dnn):
    correct = 0
    total = 0
    test_loader = torch.utils.data.DataLoader(dataset=torch.tensor(datos, dtype=torch.float),
                                              batch_size=datos.shape[0], shuffle=False)
    for data in test_loader:
        print("Tamaño del test_loader {}".format(data.shape))
        salidas = torch.tensor(data[:,-1], dtype=torch.float, requires_grad=True)
        entradas = torch.tensor(data[:,:-1], dtype=torch.float, requires_grad=True)
        outputs = dnn(entradas)
        _, predicted = torch.max(outputs.data, 1)
        print("Tamaño del predicted {}".format(predicted.shape))
        total += salidas.size(0)
        correct += (predicted == salidas.long()).sum()

    print('Accuracy of the network on the data: {}'.format(float((100 * correct / total))))

def ObtenerPrediccion(datos, dnn):
    data = torch.tensor(datos, dtype=torch.float, requires_grad=True)
    print("Shape de data: {}".format(data.shape))
    outputs = dnn(data)
    _, predicted = torch.max(outputs.data, 1)
    print("Shape de predicted {}".format(predicted.shape))
    return (predicted.numpy())
