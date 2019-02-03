# coding: utf-8
"""
Created on Sun Aug 26 15:12:14 2018

@author: Diego
"""
# Carga de librerias
import numpy as np
from os import path, getcwd, listdir
import funciones
import pickle
import funcionLSTM
import funcionesDataSet
import pandas as pd

# Carga de librerias Pytorch
import torch
from sklearn.metrics import confusion_matrix

# variables globales
version = 'LSTM Binary Classificator'
freq = 400
pathTrain = '/home/diego/Documents/DataChallenge/training/'
pathTest = '/home/diego/Documents/DataChallenge/testing/'
pathPrediccion = '/home/diego/Documents/DataChallenge/'
cwd = getcwd()
inZ = 4  # 4 lee el modulo, 1 lee eje X
inX = 1
numPacientes = 4
entrenar, predecir = (False, False)
existeVariableTrain, existeVariableTest = (False, False)
formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m',
}
batchSize = 2400  # debe ser multiplo de 6

# COMIENZO
print('{GREEN}SEIZSAFE LSTM BINARY CLASSIFICATOR{END}'.format(**formatters))
print('__________________________________')
print()
print()

while True:
    accion = input('Que quieres hacer? [E]ntrenar|[P]redecir: ')
    if accion in ('E', 'e'):
        entrenar = True
        break
    elif accion in ('P', 'p'):
        predecir = True
        break

if entrenar:
    while True:
        existeVariableTrain = input('Existe archivo de variables de train? [S]i|[N]o: ')
        if existeVariableTrain in ('S', 's'):
            existeVariableTrain = True
            print('Ruta de las variables [{}].'.format(cwd))
            rutaVariablesTrain = input()
            rutaVariablesTrain = path.join(cwd, rutaVariablesTrain)
            print(rutaVariablesTrain)
            break
        elif existeVariableTrain in ('N', 'n'):
            existeVariableTrain = False
            print("Ruta de los archivos de train [{}]".format(pathTrain))
            while True:
                correcto = input("Correcto? [S]i|[N]o ")
                if correcto in ('S', 's'):
                    break
                elif correcto in ('N', 'n'):
                    pathTrain = path.abspath(input("Nueva ruta: "))
                    break
            break

if predecir:
    while True:
        print("Ruta del modelo [{}].".format(cwd))
        rutaModelo = input()
        rutaModelo = path.join(cwd, rutaModelo)
        if path.isfile(rutaModelo):
            break
        else:
            print("{RED}La ruta no es valida{END}".format(**formatters))

    while True:
        existeVariableTest = input("Existe archivo de variables de test? [S]i|[N]o")
        if existeVariableTest in ('S', 's'):
            print("Ruta de las variables [{}] : ".format(cwd))
            rutaVariablesTest = input()
            rutaVariablesTest = path.join(cwd, rutaVariablesTest)
            if path.isfile(rutaVariablesTest):
                existeVariableTest = True
                break
            else:
                print("{RED}La ruta no es valida{END}".format(**formatters))

        elif existeVariableTest in ('N', 'n'):
            existeVariableTest = False
            print("Ruta de los archivos de test [{}]".format(pathTest))
            while True:
                correcto = input("Correcto? [S]i|[N]o ")
                if correcto in ('S', 's'):
                    break
                elif correcto in ('N', 'n'):
                    pathTest = path.abspath(input("Nueva ruta: "))
                    break


# La idea es crear una variable que almacene las muestras en el siguiente formato
# (6, numero de casos, 18)
# 6 cada fila representa un instante temporal
# numero de casos: uno por cada 6*15 segundos de archivo
# 18 es el numero de variables a analizar
# 1: [V1z, V1x], [V2z, V2x],...,[Vnz, V1x] = (numeroVentanas * 18)

if entrenar and not existeVariableTrain:
    variablesTrain = None
    totalVentanas = 0
    listaArchivos = listdir(pathTrain)

    # Por cada archivo leer el contenido del csv
    for archivo in listaArchivos:
        print(archivo)
        # Si no es archivo continuo, evito . y ..
        if not path.isfile(path.join(pathTrain, archivo)):
            continue
        # Leer datos
        datosM, datosX, longitudDatos = funciones.leer_csv_2in(path.abspath(path.join(pathTrain, archivo)), inZ, inX)
        # Comprobar longitud de los archivos
        if longitudDatos == 15 * freq:
            pass
        elif longitudDatos > 15 * freq:  # se recorta
            longitudEntera = longitudDatos // (15 * freq)
            longitudEntera = 60 if longitudEntera > 60 else longitudEntera
            datosM = datosM[0: longitudEntera * (15 * freq)]
            datosX = datosX[0: longitudEntera * (15 * freq)]
        elif longitudDatos < 15 * freq:  # se amplia y luego se recorta
            numeroRepeticiones = ((15 * freq) // longitudDatos) + 1
            datosM = np.tile(datosM, numeroRepeticiones)
            datosX = np.tile(datosX, numeroRepeticiones)
            longitudEntera = datosM.shape[0] // (15 * freq)
            datosM = datosM[0: longitudEntera * (15 * freq)]
            datosX = datosX[0: longitudEntera * (15 * freq)]

        # Evaluamos siempre por si hay error
        assert (datosM.shape[0] == datosX.shape[0]), 'Shape de de datosM y DatosX no son lo mismo'
        assert (datosM.shape[0] % (2.5 * freq) == 0), 'Longitud de los datos no es multiplo de 15 sg'
        longitudDatos = datosM.shape[0]

        # Hacer ventanas
        numeroVentanas = int(longitudDatos // (2.5 * freq))
        ventanas = funciones.calcularventana(numeroVentanas, freq)
        assert (numeroVentanas % 6 == 0), 'Numero de ventanas no es multiplo de 6'

        totalVentanas += numeroVentanas
        # Filtrado de la señal
        from scipy import signal

        b, a = signal.butter(4, 25 / freq / 2, 'low')
        datosM = signal.filtfilt(b, a, datosM)
        datosX = signal.filtfilt(b, a, datosX)

        variableTemporal = np.array([])
        for j in range(numeroVentanas):
            inicio = int(ventanas[j, 0])
            fin = int(ventanas[j, 1])
            datosTrabajoM = datosM[inicio: fin + 1]
            datosTrabajoX = datosX[inicio: fin + 1]
            # 0 -> Energia, 1 -> ACF, 2 -> FFT0to8, 3-> FFT8to16, 4 -> FFT16to30
            # 5 -> EE, 6-> kur, 7-> skew, 8 -> var

            energia = funciones.calcularenergia(datosTrabajoM)
            acf = funciones.autocorr(datosTrabajoM)
            FFT0to8, FFT8to16, FFT16to30, EE = funciones.calcularfftyee(datosTrabajoM, freq)
            kur, skew, var = funciones.calcularestadisticos(datosTrabajoM)
            variableTemporal = np.append(variableTemporal,
                                         [energia, acf, FFT0to8, FFT8to16, FFT16to30, EE, kur, skew, var])

            energia = funciones.calcularenergia(datosTrabajoX)
            acf = funciones.autocorr(datosTrabajoX)
            FFT0to8, FFT8to16, FFT16to30, EE = funciones.calcularfftyee(datosTrabajoX, freq)
            kur, skew, var = funciones.calcularestadisticos(datosTrabajoX)
            variableTemporal = np.append(variableTemporal,
                                         [energia, acf, FFT0to8, FFT8to16, FFT16to30, EE, kur, skew, var,
                                          (0, 1)['ATQ' in archivo], int(funciones.devolverpaciente(archivo))])
            assert variableTemporal.shape[0] % 20 == 0, "Variable temporal no es multiplo de 20"

        # Montar la variable
        assert variableTemporal.shape[0] % (6 * 20) == 0, 'variableTemporal no es multiplo de 120'
        variableLocal = np.zeros((6, int(numeroVentanas / 6), 20))
        indiceFilas, indiceColumna, indiceZ = (-1, -1, 0)
        for indice, valor in enumerate(variableTemporal, 0):
            if indice % 20 == 0:
                indiceFilas += 1
                if indiceFilas == 6:
                    indiceFilas = 0

            if indice % 120 == 0:
                indiceColumna += 1

            indiceZ = indice % 20
            variableLocal[indiceFilas, indiceColumna, indiceZ] = valor

        if variablesTrain is None:  # es la primera vez y variables debe ser inicializado
            assert all((variableLocal.shape[0] == 6,
                        variableLocal.shape[2] == 20)), \
                        'No coinciden los tipos de las variables. Inicializando variables'
            variablesTrain = variableLocal
        else:
            assert all((variableLocal.shape[0] == variablesTrain.shape[0],
                        variableLocal.shape[2] == variablesTrain.shape[2])), 'No coinciden los tipos de las matrices'
            variablesTrain = np.append(variablesTrain, variableLocal, axis=1)

    print('Variables de Train. Resultado: {}'.format(variablesTrain.shape))
    print("Numero de ataques: {}".format(variablesTrain[variablesTrain[:, :, 18] == 1].shape[0] / 6))
    print("Numero de muestras de paciente 1 {}".format(variablesTrain[variablesTrain[:, :, 19] == 1].shape))
    print("Numero total de ventanas: {}".format(totalVentanas))
    print("------------------------------------------------------------")
    # Guardar variables en pickle
    while True:
        correcto = input("Guardar variables de train? [S]i|[N]o")
        if correcto in ('S', 's'):
            rutaVariablesTrain = input("Archivo: Ruta[{}] : ".format(cwd))
            rutaVariablesTrain = path.abspath(path.join(getcwd(), rutaVariablesTrain))
            diccionario = {'VARIABLES': variablesTrain}
            pickle.dump(diccionario, open(rutaVariablesTrain, "wb"))
            print("Variables guardadas en el archivo: {}".format(rutaVariablesTrain))
            break
        if correcto in ('N', 'n'):
            break

if entrenar and existeVariableTrain:
    diccionario = pickle.load(open(rutaVariablesTrain, 'rb'))
    variablesTrain = diccionario['VARIABLES']
    print("He cargado la variables de train. Resultado: {}".format(variablesTrain.shape))
    print("Numero de movimientos: {}".format(variablesTrain[variablesTrain[:, :, 18] == 1].shape[0]/6))
    print("------------------------------------------------------------")


if predecir and not existeVariableTest:
    variablesTest = None
    # totalVentanas = 0
    listaArchivos = listdir(pathTest)

    # Por cada archivo leer el contenido del csv
    for indiceArchivo, archivo in enumerate(listaArchivos):
        print(archivo)
        # Si no es archivo continuo, evito . y ..
        if not path.isfile(path.join(pathTest, archivo)):
            continue
        # Leer datos
        datosM, datosX, longitudDatos = funciones.leer_csv_2in(path.abspath(path.join(pathTest, archivo)), inZ, inX)
        # Comprobar longitud de los archivos
        if not longitudDatos == 15 * freq:
            print("{RED}Hay un archivo que no tiene longitud de 15 segundos{END}".format(**formatters))
            exit(2)

        # Evaluamos siempre por si hay error
        assert (datosM.shape[0] == datosX.shape[0]), 'Shape de de datosM y DatosX no son lo mismo'
        assert (datosM.shape[0] == 15 * freq), 'Longitud de los datos no es multiplo de 15 sg'

        # Hacer ventanas
        numeroVentanas = int(longitudDatos // (2.5 * freq))
        ventanas = funciones.calcularventana(numeroVentanas, freq)
        assert (numeroVentanas == 6), '{RED}Numero de ventanas debe ser 6{END}'.format(**formatters)

        # Filtrado de la señal
        from scipy import signal

        b, a = signal.butter(4, 25 / freq / 2, 'low')
        datosM = signal.filtfilt(b, a, datosM)
        datosX = signal.filtfilt(b, a, datosX)

        variableTemporal = np.array([])
        for j in range(numeroVentanas):
            inicio = int(ventanas[j, 0])
            fin = int(ventanas[j, 1])
            datosTrabajoM = datosM[inicio: fin + 1]
            datosTrabajoX = datosX[inicio: fin + 1]
            # 0 -> Energia, 1 -> ACF, 2 -> FFT0to8, 3-> FFT8to16, 4 -> FFT16to30
            # 5 -> EE, 6-> kur, 7-> skew, 8 -> var

            energia = funciones.calcularenergia(datosTrabajoM)
            acf = funciones.autocorr(datosTrabajoM)
            FFT0to8, FFT8to16, FFT16to30, EE = funciones.calcularfftyee(datosTrabajoM, freq)
            kur, skew, var = funciones.calcularestadisticos(datosTrabajoM)
            variableTemporal = np.append(variableTemporal,
                                         [energia, acf, FFT0to8, FFT8to16, FFT16to30, EE, kur, skew, var])

            energia = funciones.calcularenergia(datosTrabajoX)
            acf = funciones.autocorr(datosTrabajoX)
            FFT0to8, FFT8to16, FFT16to30, EE = funciones.calcularfftyee(datosTrabajoX, freq)
            kur, skew, var = funciones.calcularestadisticos(datosTrabajoX)
            variableTemporal = np.append(variableTemporal,
                                         [energia, acf, FFT0to8, FFT8to16, FFT16to30, EE, kur, skew, var,
                                          indiceArchivo, int(funciones.devolverpaciente(archivo))])
            assert variableTemporal.shape[0] % 20 == 0, "Variable temporal no es multiplo de 20"

        # Montar la variable
        assert variableTemporal.shape[0] % (6 * 20) == 0, 'variableTemporal no es multiplo de 120'
        variableLocal = np.zeros((6, int(numeroVentanas / 6), 20))
        indiceFilas, indiceColumna, indiceZ = (-1, -1, 0)
        for indice, valor in enumerate(variableTemporal, 0):
            if indice % 20 == 0:
                indiceFilas += 1
                if indiceFilas == 6:
                    indiceFilas = 0

            if indice % 120 == 0:
                indiceColumna += 1

            indiceZ = indice % 20
            variableLocal[indiceFilas, indiceColumna, indiceZ] = valor

        if variablesTest is None:  # es la primera vez y variables debe ser inicializado
            assert all((variableLocal.shape[0] == 6,
                        variableLocal.shape[2] == 20)), \
                        'No coinciden los tipos de las variables. Inicializando variables'
            variablesTest = variableLocal
        else:
            assert all((variableLocal.shape[0] == variablesTest.shape[0],
                        variableLocal.shape[2] == variablesTest.shape[2])), 'No coinciden los tipos de las matrices'
            variablesTest = np.append(variablesTest, variableLocal, axis=1)

    numeroArchivos = variablesTest.shape[0] / 6
    print('Variables de Test. Resultado: {}'.format(variablesTest.shape))
    print('Numero de archivos: {}'.format(variablesTest.shape[0] / 6))
    print("------------------------------------------------------------")
    # Guardar variables en pickle
    while True:
        correcto = input("Guardar variables de test? [S]i|[N]o ")
        if correcto in ('S', 's'):
            rutaVariablesTest = input("Archivo: Ruta[{}] : ".format(cwd))
            rutaVariablesTest = path.abspath(path.join(getcwd(), rutaVariablesTest))
            diccionario = {'VARIABLES': variablesTest, 'ARCHIVOS': variablesTest.shape[0] / 6}
            pickle.dump(diccionario, open(rutaVariablesTest, "wb"))
            print("Variables guardadas en el archivo: {}".format(rutaVariablesTest))
            break
        if correcto in ('N', 'n'):
            break

if predecir and existeVariableTest:
    diccionario = pickle.load(open(rutaVariablesTest, 'rb'))
    variablesTest = diccionario['VARIABLES']
    numeroArchivos = diccionario['ARCHIVOS']
    print("He cargado la variables de train. Resultado: {}".format(variablesTest.shape))
    print("Numero de archivos:{}".format(variablesTest.shape[0] / 6))
    print("------------------------------------------------------------")

if entrenar:
    import torch
    import torch.autograd as autograd
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable

    numEpoch = 100
    learning_rate = 0.01
    lossTraining = np.array([])
    lossValidation = np.array([])
    accuracy = np.array([])
    minAccuracy = 0.0

    variableModeloK = variablesTrain.reshape(6, -1, 20)
    variableModeloK = variableModeloK[:, :, :-1]  # quitamos paciente

    dataset = funcionesDataSet.Data3DSet(variableModeloK)
    dataset.multiplicar_ataques()
    escalador = dataset.normalizacion()

    diccionario = {'SCALER': escalador}
    pickle.dump(diccionario, open(path.join(cwd, "Escalador.pickle"), "wb"))

    dataset.shuffle()
    dataset.split(batch_size=batchSize, porcentaje_train=0.8)
    classWeights = dataset.weights_clases()

    # Crear red
    miLstm = funcionLSTM.LSTM(input_dim=18, batch_size=int(batchSize), hidden_dim=24, output_dim=2, num_layers=2, dropout=0.5)
    lossFN = nn.CrossEntropyLoss(weight=classWeights)
    optimiser = optim.Adam(miLstm.parameters(), lr=learning_rate)

    for epoch in range(numEpoch):
        miLstm.hidden = miLstm.init_hidden()
        for i in range(int(dataset.__len__(train=True, test=False) / batchSize)):
            rangoBatch = range(i * batchSize, (i + 1) * batchSize, 1)
            data = torch.tensor(dataset.__getitem__(rangoBatch, train=True, test=False))
            # print("EVALUACION TRAIN")   
            # for t in range(data.shape[1]):
            #    assert data[0, t, -1] == data[1, t, -1] == data[2, t, -1] == data[3, t, -1] \
            #        == data[4, t, -1] == data[5, t, -1] , "No son iguales"

            # print("data.shape: {}".format(data.shape))
            y_train = Variable(data[0, :, -1]).long()
            # print("y_train.shape: {}".format(y_train.shape))

            x_train = Variable(data[:, :, :-1]).float()
            # print("x_train.shape: {}".format(x_train.shape))

            optimiser.zero_grad()
            y_pred = miLstm(x_train)
            # print("y_pred.shape: {}".format(y_pred.shape))

            loss = lossFN(y_pred, y_train)

            loss.backward()
            optimiser.step()

        lossTraining = np.append(lossTraining, loss.item())
        print('Epoch [%d/%d], Loss: %.4f' % (epoch, numEpoch, loss.item()))

        if epoch % 5 == 0:
            lossVal = 0.0
            accuracyVal = 0.0
            for j in range(int(dataset.__len__(train=False, test=True) / batchSize)):
                # convertir de 2D a 3D
                # dataIn = dataIn.reshape(6, -1, 19)
                rangoBatch = range(j * batchSize, (j + 1) * batchSize, 1)
                dataIn = torch.tensor(dataset.__getitem__(rangoBatch, train=False, test=True))
                # a = dataIn
                # print("EVALUACIONi VALIDATION")   
                # for t in range(a.shape[1]):
                #    assert a[0, t, -1] == a[1, t, -1] == a[2, t, -1] == a[3, t, -1] \
                #       == a[4, t, -1] == a[5, t, -1] , "No son iguales"

                entradaRed = torch.tensor(dataIn[:, :, :-1]).float()
                # print("entradaRed.shape: {}".format(entradaRed.shape))
                miLstm.eval()
                y_validationPred = miLstm(entradaRed)
                miLstm.train()
                _, salidas = torch.max(y_validationPred, 1)
                lossVal += lossFN(y_validationPred, torch.tensor(dataIn[0, :, -1]).long()).item()
                # print("salida.shape : {} , dataIn.shape: {}".format(y_validationPred.shape, dataIn[0, :, -1].flatten().shape))
                tn, fp, fn, tp = confusion_matrix(dataIn[0, :, -1], salidas, labels=[0, 1]).ravel()
                accuracyVal += 55 * (tp / (tp + fn)) + 42.5 * (tn / (tn + fp))

                # accuracyVal += accuracy_score(salidas, dataIn[0, :, -1].long())
                # print("accuracy: {}".format(accuracy_score(salidas, dataIn[0, : , -1].long())))

            accuracyVal /= (j+1)
            print('Epoch [%d/%d], LossValidacion: %.4f Accuracy %.6f' % (epoch, numEpoch, lossVal / (j + 1), accuracyVal))
            # print("j es: {}".format(j))
            lossValidation = np.append(lossValidation, lossVal / (j + 1))
            accuracy = np.append(accuracy, accuracyVal)

            if accuracyVal > minAccuracy:
                minAccuracy = accuracyVal
                torch.save(miLstm, path.join(cwd, "LTSMUnica" + ".pt"))

if predecir:
    print("Predecir {} archivos...".format(numeroArchivos))
    variableModeloK = variablesTest.reshape(6, -1, 20)
    variableModeloK = variableModeloK[:, :, :-1]  # quitamos paciente

    # La variable debe ser multiplo del batchSize
    batchEntero = variableModeloK.shape[1] // batchSize
    batchEntero = batchSize * (batchEntero + 1) - variableModeloK.shape[1]
    complemento = np.ones((6, batchEntero, 19))
    variableModeloK = np.append(variableModeloK, complemento, axis=1)
    assert(variableModeloK.shape[1] % batchSize == 0), "El dataset no tiene muestras multiplo de batchSize"
    print("variableModeloK.shape: {}".format(variableModeloK.shape))

    datasetProduccion = funcionesDataSet.Data3DSet(variableModeloK)
    diccionario = pickle.load(open(path.join(cwd, "Escalador.pickle"), 'rb'))
    _ = datasetProduccion.normalizacion(diccionario['SCALER'])
    # datasetProduccion.split(batchSize, porcentaje_train=1)
    # entradas = datasetProduccion.dataTrain
    # print("entradas.shape: {}".format(entradas.shape))

    # cargar la red neuronal
    miLstm = torch.load(rutaModelo)
    miLstm.eval()
    identificadores = []
    salidasModelo = []
    for i in range(int(datasetProduccion.__len__(train=False, test=False) / batchSize)):
        print("\tprediciendo archivo {} a {}".format(i * batchSize, (i + 1) * batchSize), end="")
        rangoBatch = range(i * batchSize, (i + 1) * batchSize, 1)
        data = torch.tensor(datasetProduccion.__getitem__(rangoBatch, train=False, test=False))
        identificador = data[:, :, -1]
        entradaModelo = Variable(data[:, :, :-1]).float()
        salidaModelo = miLstm(entradaModelo)
        identificadores = np.append(identificadores, identificador.to_nu)
        salidasModelo = np.append(salidasModelo, salidaModelo)
        print("\t{GREEN]Correcto{END}".format(**formatters))

    print("salidasModelo.shape".format(salidasModelo.shape))
    print("identificadores.shape".format(identificadores.shape))
    salidasModelo = salidasModelo[0:numeroArchivos]
    identificadores = identificadores[0:numeroArchivos]
    print("salidasModelo.shape".format(salidasModelo.shape))
    print("identificadores.shape".format(identificadores.shape))
    # montar el dataframe
    df = pd.DataFrame(np.append(identificadores, salidasModelo).reshape(-1, 2), columns=["ARCHIVO", "PREDICCION"])



