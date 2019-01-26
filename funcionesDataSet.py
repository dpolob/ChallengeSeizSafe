"""
    Coge el dataset en bruto y le aplica las siguientes operaciones
"""

from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


class Data3DSet(Dataset):
    def __init__(self, data):
        super(Data3DSet, self).__init__()
        self.formatters = {
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'END': '\033[0m',
        }
        self.data = data
        self.dataTrain = []
        self.dataTest = []
        print("Creacion del dataset")
        print("\tTamaño del dataset: {}".format(self.data.shape))
        print("{GRENN}Correcto{END}".format(**self.formatters))

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index, train=False, test=False):
        if not train and not test:
            return self.data[:, index, :]
        if train:
            return self.dataTrain[:, index, :]
        if test:
            return self.dataTest[:, index, :]

    def comprobacion(self, entrada):
        """
        Comprueba que la matriz resultante esta ordenada
        :return: Si error, lanza una excepcion
        """

        for t in range(entrada.shape[1]):
            assert entrada[0, t, -1] == entrada[1, t, -1] == entrada[2, t, -1] == \
                   entrada[3, t, -1] == entrada[4, t, -1] == entrada[5, t, -1], \
                   "{RED}Error en la comprobacion del shuffle{END}".format(**self.formatters)

    def shuffle(self):
        """
        Hace un shuffle del dataset en la dimension 1 [6, muestras, 19]
        :return: el dataset aleatorio
        """
        aleatorio = np.arange(self.data.shape[1])
        np.random.shuffle(aleatorio)
        print("Shuffle del dataset")
        print("\tentrada: {}".format(self.data.shape))
        self.data = self.data[:, aleatorio, :]
        print("\tsalida: {}".format(self.data.shape))
        self.comprobacion(self.data)
        print("{GRENN}Correcto{END}".format(**self.formatters))
        
    def split(self, batch_size, porcentaje_train=0.8):
        """
        Divide el dataset en train y test segun el porcentaje dado. Normaliza siempre a multiplos del
        batch
        :param batch_size: tamaño del batch
        :param porcentaje_train: % de muestras en train
        """
        print("Split del dataset")
        print("\tentrada: {}". format(self.data.shape))
        muestrasTrain = (self.data.shape[1] * porcentaje_train) // batch_size
        muestrasTrain = int(muestrasTrain * batch_size)
        muestrasTest = (self.data.shape[1] - muestrasTrain) // batch_size
        muestrasTest = int(muestrasTest * batch_size)
        print("\tbatchSize: {}".format(batch_size))
        print("\ttamaño del set de train: {}".format(muestrasTrain))
        print("\ttamaño del set de test: {}".format(muestrasTest))
        self.dataTrain = self.data[:, muestrasTrain, :]
        self.comprobacion(self.dataTrain)
        self.dataTest = self.dataTrain[:, muestrasTrain:muestrasTrain + muestrasTest, :]
        self.comprobacion(self.dataTest)
        print("{GRENN}Correcto{END}".format(**self.formatters))

    def normalizacion(self):
        """
        Normaliza el dataset
        """
        print("Normalizcion")
        scaler = StandardScaler()
        self.data[:, :-1] = scaler.fit_transform(self.data[:, :-1])
        self.data = self.data.reshape(6, -1, 19)
        self.comprobacion(self.data)
        print("{GREEN}Correcto{END}".format(**self.formatters))

    def multiplicar_ataques(self):
        print("Multiplicar ataques")
        ataques = self.data[self.data[:, :, -1] == 1].reshape(6, -1, 19)
        print("\tNumero de ataques originales : {}".format(ataques.shape[1]))
        numeroAtaques = ataques.shape[1]
        # ataques = np.repeat(ataques, int(variableModeloK.shape[1] // numeroAtaques) - 1, axis=1)
        ataques = np.repeat(ataques, int((self.data.shape[1] // numeroAtaques) / 4), axis=1)
        self.data = np.append(self.data, ataques, axis=1)
        print("\tNumero de ataques despues de la multiplicacion: {}"
              .format(self.data[self.data[:, :, -1] == 1].shape[0] / 6))
        self.comprobacion(self.data)
        print("{GRENN}Correcto{END}".format(**self.formatters))

    def weights_clases(self):
        print("Calcular pesos de las clases")
        ataques = self.dataTrain[self.dataTrain[:, :, -1] == 1].reshape(6, -1, 19)
        numeroAtaques = ataques.shape[1]
        classWeights = torch.tensor([1 / (self.dataTrain.shape[1] - numeroAtaques), 1 / numeroAtaques])
        print("\tClase 0: {}".format(1 / (self.dataTrain.shape[1] - numeroAtaques)))
        print("\tClase 1: {}".format(1 / numeroAtaques))
        return classWeights
