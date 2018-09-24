# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:12:14 2018

@author: Diego
"""
#Carga de librerias
import pdb
import numpy as np
import pandas as pd
from os import path, getcwd, listdir
from sklearn import utils
import funciones
import getopt
import sys
import pickle



#variables globales
version = '3.0'
freq = 400
cwd = getcwd()
inCsv = 4 # 4 lee el modulo, 3 lee eje Z
primeraVez = True
rutaPickle = None
rutaPickleS = None


#inVar = np.array([0,1,3,4,5,7])


print("Frecuencia: {}\n".format(freq))
print("CWD       : {}\n".format(cwd))

options, remainder = getopt.getopt(sys.argv[1:], 'e:s:v:p:ps', ['version', 'entrenar=', 'sensibilidad=', 'valorar=', 'pickle=', 'pickles='])
training, sensibilidad, valorar, modoPickle, modoPickleS = (False, False, False, False, False) 

#Leer linea de comandos del programa
for opt, arg in options:
    if opt in ('-e', '--entrenar'):
        training = True
        normalizedPathEntrenar = path.abspath(arg)
    elif opt in ('-s', '--sensibilidad'):
        sensibilidad = True
        normalizedPathSensibilidad = path.abspath(arg)
    elif opt in ('-v','--valorar'):
        valorar = True
        archivoSalida = path.abspath(arg)
    elif opt in ('-p', '--pickle'):
        rutaPickle = path.abspath(arg)
        modoPickle = True
    elif opt in ('-ps', '--pickles'):
        rutaPickleS = path.abspath(arg)
        modoPickleS = True
    elif opt == '--version':
        print(version)
        exit(0)
    else:
       exit(2)



if training:    
    if modoPickle == False:    
        variables = None
        salidas = None

        listaArchivos = listdir(normalizedPathEntrenar)
    
        #Por cada archivo leer el contenido del csv
        for archivo in listaArchivos:
            print(archivo + "\n")
        
            #SI NO ES ARCHIVO SALTO
            if path.isfile(path.join(normalizedPathEntrenar, archivo)) == False:
                continue
       
            datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(normalizedPathEntrenar , archivo)), inCsv)

            #Comprobar longitud de los archivos
            if longitudDatos == 15 * freq:
                numeroVentanas = 6
            elif longitudDatos > 15 * freq:
                numeroVentanas = 6
                datos = datos[0 : (15 * freq)] #ojo que es [ : )
                longitudDatos = int(15 * freq)
            elif longitudDatos < 15 * freq:
                numeroVentanas = int(longitudDatos // (2.5 * freq))
                longitudDatos = int(2.5 * numeroVentanas * freq)
                datos = datos[0 : longitudDatos]
            
            #Comprobar longitud de los archivos
            #if 'ATQ' in archivo: #Si es un ataque entonces se procesa entero
                #if longitudDatos == 15 * freq:
                    #numeroVentanas = 6
                #elif longitudDatos > 15 * freq:
                    #numeroVentanas = 6
                    #datos = datos[0 : (15 * freq)] #ojo que es [ : )
                    #longitudDatos = int(15 * freq)
                #elif longitudDatos < 15 * freq:
                    #numeroVentanas = int(longitudDatos // (2.5 * freq))
                    #longitudDatos = int(2.5 * numeroVentanas * freq)
                    #datos = datos[0 : longitudDatos]
            #elif 'MOV' in archivo: # Si es un movimiento solo se procesa una ventana si 6000<long<24000
                #if (longitudDatos < 24000 and longitudDatos > 6000):
                    #numeroVentanas = 1
                    #datos = datos[0 : int(2.5 * freq)]
                    #longitudDatos = datos.shape[0]
                #else:
                    #continue



            #calcular datos de inicio y fin de las ventanas
            ventanas = funciones.calcularventana(numeroVentanas, freq)
            #calcular ventana hamming para filtrado de extremos
            hammingWindow = funciones.hamming(longitudDatos, freq)
            #rehacer datos con la ventana hamming
            datos = datos * hammingWindow
            #inicializo la ventana a ceros
            variableLocal = np.zeros((numeroVentanas, 13))
        
            if "ATQ" in archivo:
                salidaLocal = np.ones(numeroVentanas, dtype=int)
            else:
                salidaLocal = np.zeros(numeroVentanas, dtype=int)
          
            for j in range(numeroVentanas):
                inicio = int(ventanas[j,0])
                fin = int(ventanas[j,1])
                datosTrabajo = datos[inicio : fin + 1]
        
                #0 -> Energia,         #1 -> RMS,         #2 -> Vpp,         #3 -> ACF
                #4 -> FFT025,         #5 -> FFT25100,         #6 -> FFT100200
                #7 -> EE, 8-> kur 9-> skew , 10 -> var, 11 -> entropia
               
                variableLocal[j, 0] = funciones.calcularenergia(datosTrabajo)
                variableLocal[j, 1] = funciones.calcularRMS(datosTrabajo)
                variableLocal[j, 2] = funciones.calcularVpp(datosTrabajo)
                variableLocal[j, 3] = funciones.autocorr(datosTrabajo)
                variableLocal[j, 4], variableLocal[j, 5], variableLocal[j, 6], variableLocal[j, 7]  = funciones.calcularfftyee(datosTrabajo, freq)
                variableLocal[j, 8], variableLocal[j, 9], variableLocal[j, 10] = funciones.calcularestadisticos(datosTrabajo)
                variableLocal[j, 11] = funciones.calcularentropia(datosTrabajo)
                variableLocal[j, 12] = funciones.devolverpaciente(archivo)

            if variables is None: #es la primera vez y variables y salida debe ser inicializado
                variables = variableLocal
                salidas = salidaLocal
            else:
                variables = np.vstack((variables, variableLocal))
                salidas = np.append(salidas, salidaLocal)
        

        while True:
            respuestaUsuario = input("Desea guardar los resultados de las variables/salidas? (y/n)\n")
            if (respuestaUsuario == 'y' or respuestaUsuario == 'n'):
                break

        if respuestaUsuario == 'y':
           if rutaPickle is None:
               rutaPickle = path.abspath(path.join(getcwd(),"variables.pickle"))
           diccionario = {'VARIABLES' : variables, 'SALIDAS' : salidas}        
           pickle.dump(diccionario, open(rutaPickle, "wb" ))
           print("Variables guardadas en el archivo: {}\n".format(rutaPickle))

    elif modoPickle == True:
        diccionario = pickle.load(open(rutaPickle, 'rb'))
        variables = diccionario['VARIABLES']
        salidas = diccionario['SALIDAS']
    
    #Aqui ya tengo las variables en variables y salidas ya sea por pick o analisis
    print ("------------------------------------------------------\n")
    print ("Ya he leido todos los datos, con las condiciones impuestas\n")
    print (" * Numero de ataques: {}\n".format(sum(x for x in salidas if x==1)))
    print (" * Numero de movimientos: {}\n".format(salidas.shape[0] - sum(x for x in salidas if x==1)))
    tirar = input("Pulse una tecla para entrenar...\n")

    #ENTRENAMOS UN MODELO SVM con CV con gridsearch
    #    from sklearn.model_selection import train_test_split
    #    X_train, X_test, y_train, y_test = train_test_split(variables[:,:], salidas, test_size=0.50, random_state=42, shuffle=True)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    #Haremos 4 SVM uno por cada ataque
    variableCombinada = np.concatenate((variables, salidas.reshape(len(salidas),1)), axis=1)
    print(variableCombinada)
    tirar = input("Pulse una tecla para entrenar...\n")
    modelos = dict()

    for k in range(1, 5):
        variableCombinadaModeloK = variableCombinada[ variableCombinada[:, 12] == k ]
        entradaModeloK = variableCombinadaModeloK[:, :-2]
        print("Ciclo: {}".format(k))
        print("Shape EntradaModeloK: {}".format(entradaModeloK.shape))
        salidaModeloK = variableCombinadaModeloK[:, -1]
        print("Shape SalidaModeloK: {}".format(salidaModeloK.shape))
        print("Numero de ataques: {}".format(sum(salidaModeloK[salidaModeloK == 1])))
        scalerK = StandardScaler().fit(entradaModeloK)
        entradaModeloK = scalerK.transform(entradaModeloK)
        parametersK = {'kernel':('rbf', 'sigmoid', 'poly'), 'C':[0.01, 0.1, 1, 10, 100, 1000, 10000]}
        svcK = svm.SVC()
        clfK = GridSearchCV(svcK, parametersK)
        clfK.fit(entradaModeloK, salidaModeloK)
        modelos["SCALER" + str(k)] = scalerK
        modelos["SVC" + str(k)] = clfK.best_estimator_

#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
        
#    parameters = {'kernel':('rbf', 'sigmoid', 'poly'), 'C':[0.01, 0.1, 1, 10, 100, 1000]}
#    svc = svm.SVC()
#    clf = GridSearchCV(svc, parameters)
#    clf.fit(variables, salidas)
    print ("------------------------------------------------------")
    print ("Ya he entrenado {} modelos".format(k))
    tirar = input("Pulse una tecla para evaluar...")

    #Una vez entranado calculamos como saldría el resultado con el dataset de sensibilidad
    if modoPickleS == False
        listaArchivosSensibilidad = listdir(normalizedPathSensibilidad)
        variables = None
        salidas = None
        for archivo in listaArchivosSensibilidad:
            print(archivo + "\n")
            
            #SI NO ES ARCHIVO SALTO
            if path.isfile(path.join(normalizedPathSensibilidad, archivo)) == False:
                continue
           
            datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(normalizedPathSensibilidad , archivo)), inCsv)

            #Comprobar longitud de los archivos
            
            if longitudDatos == 15 * freq:
                numeroVentanas = 6
            elif longitudDatos > 15 * freq:
                numeroVentanas = 6
                datos = datos[0 : (15 * freq)] #ojo que es [ : )
                longitudDatos = int(15 * freq)
            elif longitudDatos < 15 * freq:
                numeroVentanas = int(longitudDatos // (2.5 * freq))
                longitudDatos = int(2.5 * numeroVentanas * freq)
                datos = datos[0 : longitudDatos]
           
            #calcular datos de inicio y fin de las ventanas
            ventanas = funciones.calcularventana(numeroVentanas, freq)
            #calcular ventana hamming para filtrado de extremos
            hammingWindow = funciones.hamming(longitudDatos, freq)
            #rehacer datos con la ventana hamming
            datos = datos * hammingWindow
            #inicializo la ventana a ceros
            variableLocal = np.zeros((numeroVentanas, 13))
            
            if "ATQ" in archivo:
                salidaLocal = np.ones(numeroVentanas, dtype=int)
            else:
                salidaLocal = np.zeros(numeroVentanas, dtype=int)
            
            for j in range(numeroVentanas):
                inicio = int(ventanas[j,0])
                fin = int(ventanas[j,1])
                datosTrabajo = datos[inicio : fin + 1]
            
                    #0 -> Energia,         #1 -> RMS,         #2 -> Vpp,         #3 -> ACF
                    #4 -> FFT025,         #5 -> FFT25100,         #6 -> FFT100200
                    #7 -> EE, 8-> kur 9-> skew , 10 -> var, 11 -> entropia
                   
                variableLocal[j, 0] = funciones.calcularenergia(datosTrabajo)
                variableLocal[j, 1] = funciones.calcularRMS(datosTrabajo)
                variableLocal[j, 2] = funciones.calcularVpp(datosTrabajo)
                variableLocal[j, 3] = funciones.autocorr(datosTrabajo)
                variableLocal[j, 4], variableLocal[j, 5], variableLocal[j, 6], variableLocal[j, 7]  = funciones.calcularfftyee(datosTrabajo, freq)
                variableLocal[j, 8], variableLocal[j, 9], variableLocal[j, 10] = funciones.calcularestadisticos(datosTrabajo)
                variableLocal[j, 11] = funciones.calcularentropia(datosTrabajo)
                variableLocal[j, 12] = funciones.devolverpaciente(archivo)

                if variables is None: #es la primera vez y variables y salida debe ser inicializado
                    variables = variableLocal
                    salidas = salidaLocal
                else:
                    variables = np.vstack((variables, variableLocal))
                    salidas = np.append(salidas, salidaLocal)
         
         while True:
            respuestaUsuario = input("Desea guardar los resultados de las variables/salidas de sensibilidad? (y/n)")
            if (respuestaUsuario == 'y' or respuestaUsuario == 'n'):
                break

        if respuestaUsuario == 'y':
           if rutaPickleS is None:
               rutaPickleS = path.abspath(path.join(getcwd(),"sensibilidad.pickle"))
           diccionario = {'VARIABLES' : variables, 'SALIDAS' : salidas}        
           pickle.dump(diccionario, open(rutaPickleS, "wb" ))
           print("Variables guardadas en el archivo: {}\n".format(rutaPickleS))

    elif modoPickleS == True:
        diccionario = pickle.load(open(rutaPickleS, 'rb'))
        variables = diccionario['VARIABLES']
        salidas = diccionario['SALIDAS']
    
    print ("------------------------------------------------------")
    print ("Ya he leido todos los datos")
    print (" * Numero de ataques: {}".format(sum(x for x in salidas if x==1)))
    print (" * Numero de movimientos: {}".format(salidas.shape[0] - sum(x for x in salidas if x==1)))
    tirar = input("Pulse una tecla para predecir...")

    from sklearn.metrics import accuracy_score
    variableCombinada = np.concatenate((variables, salidas.reshape(len(salidas),1)), axis=1)
    for k in range(1, 5):
        print("Prediccion con modelo {}".format(k))
        variableCombinadaModeloK = variableCombinada[ variableCombinada[:, 12] == k ]
        entradaModeloK = variableCombinadaModeloK[:, :-2]
        print("Shape entradaModelo: {}".format(entradaModeloK.shape))
        salidaModeloK = variableCombinadaModeloK[:, -1]
        print("Shape salidaModelo: {}".format(salidaModeloK.shape))
        scalerK = modelos["SCALER" + str(k)]
        scalerK.transform(entradaModeloK[:,:])
        clfK = modelos["SVC" + str(k)]
        prediccionK = clfK.predict(entradaModeloK)
        print("-------------------------------------------------\n")
        print("RESULTADO DEL ESTIMADOR {}: {}\n".format(k, accuracy_score(salidaModeloK, prediccionK)))
        
    while True:
        respuestaUsuario = input("Desea guardar los estimadores? (y/n)\n")
        if (respuestaUsuario == 'y' or respuestaUsuario == 'n'):
            break
    if respuestaUsuario == 'y':
        pathGuardarModelo = path.abspath(input("Ruta del modelo"))
        pickle.dump(modelos, open(pathGuardarModelo, "wb" ))
        print("Estimador guardado en el archivo: {}\n".format(pathGuardarModelo))

    
if sensibilidad:
#    listaArchivos = listdir(normalizedPath)
#    variables = None
#    salidas = None

    #Por cada archivo leer el contenido del csv
#    for archivo in listaArchivos:
#        print(archivo + "\n")
        
        #SI NO ES ARCHIVO SALTO
#        if path.isfile(path.join(normalizedPath, archivo)) == False:
#            continue
          
#        datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(normalizedPath , archivo)), inCsv)
        
        #Comprobar longitud de los archivos
#        if longitudDatos == 15 * freq:
#            numeroVentanas = 6
#        elif longitudDatos > 15 * freq:
#            numeroVentanas = 6
#            datos = datos[0 : (15 * freq)] #ojo que es [ : )
#            longitudDatos = int(15 * freq)
#        elif longitudDatos < 15 * freq:
#            numeroVentanas = int(longitudDatos // (2.5 * freq))
#            longitudDatos = int(2.5 * numeroVentanas * freq)
#            datos = datos[0 : longitudDatos]
#       
#      
#        #calcular datos de inicio y fin de las ventanas
#        ventanas = funciones.calcularventana(numeroVentanas, freq)
#        
#        #calcular ventana hamming para filtrado de extremos
#        hammingWindow = funciones.hamming(longitudDatos, freq)
#        
#        #rehacer datos con la ventana hamming
#        datos = datos * hammingWindow
#
#        #inicializo la ventana a ceros
#        variableLocal = np.zeros((numeroVentanas, 12))
#        
#        if "ATQ" in archivo:
#            salidaLocal = np.ones(numeroVentanas, dtype=int)
#        else:
#            salidaLocal = np.zeros(numeroVentanas, dtype=int)
#          
#        for j in range(numeroVentanas):
#            inicio = int(ventanas[j,0])
#            fin = int(ventanas[j,1])
#            datosTrabajo = datos[inicio : fin + 1]
#            
#            #0 -> Energia,         #1 -> RMS,         #2 -> Vpp,         #3 -> ACF
#            #4 -> FFT025,         #5 -> FFT25100,         #6 -> FFT100200
#            #7 -> EE, 8-> kur 9-> skew , 10 -> var, 11 -> entropia
#                   
#            variableLocal[j, 0] = funciones.calcularenergia(datosTrabajo)
#            variableLocal[j, 1] = funciones.calcularRMS(datosTrabajo)
#            variableLocal[j, 2] = funciones.calcularVpp(datosTrabajo)
#            variableLocal[j, 3] = funciones.autocorr(datosTrabajo)
#            variableLocal[j, 4], variableLocal[j, 5], variableLocal[j, 6], variableLocal[j, 7]  = funciones.calcularfftyee(datosTrabajo, freq)
#            variableLocal[j, 8], variableLocal[j, 9], variableLocal[j, 10] = funciones.calcularestadisticos(datosTrabajo)
#            variableLocal[j, 11] = funciones.calcularentropia(datosTrabajo)
#            #variableLocal[j, 12] = funciones.devolverpaciente(archivo)
#        
#        if variables is None: #es la primera vez y variables y salida debe ser inicializado
#            variables = variableLocal
#            salidas = salidaLocal
#        else:
#            variables = np.vstack((variables, variableLocal))
#            salidas = np.append(salidas, salidaLocal)
#     
#    #Cargar estimador y scaler
#    from sklearn import svm
#    from sklearn.preprocessing import StandardScaler
#    import pickle
#
#    pickle.load(diccionario, open(rutaModelo, 'rb'))
#    modelo = diccionario['SVM']
#    scaler = diccionario['SCALER']

elif valorar:
    print()
