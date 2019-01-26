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

    def __len__(self, train=False, test=False):
        if not train and not test:
            return self.data.shape[1]
        if train:
            return self.dataTrain.shape[1]
        if test:
            return self.dataTest.shape[1]

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
