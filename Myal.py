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
import DNN


#variables globales
version = '3.0'
freq = 400
cwd = getcwd()
inCsv = 4 # 4 lee el modulo, 3 lee eje Z
num_pacientes = 4
primeraVez = True
rutaPickleE = None
rutaPickleS = None
rutaPickleM = None
rutaSalida = None

#inVar = np.array([0,1,3,4,5,7])


print("Frecuencia: {}".format(freq))
print("CWD       : {}".format(cwd))

options, remainder = getopt.getopt(sys.argv[1:], 'rE:rS:e:s:v:pE:pS:pM:o:h', ['rutaE=', 'rutaS=', 'entrenar', 'sensibilidad', 'valorar', 'pickleE=', 'pickleS=', 'pickleM=', 'salida', 'help'])
rutaE, rutaS, entrenar, sensibilidad, valorar, modoPickleE, modoPickleS, modoPickleMa = (False, False, False, False, False, False, False, False)

#Leer linea de comandos del programa
for opt, arg in options:
    if opt in ('-rE', '--rutaE'):
        rutaE = path.abspath(arg)
    elif opt in ('-rS', '--rutaS'):
        rutaS = path.abspath(arg)
    elif opt in ('-e', '--entrenar'):
        entrenar = True
    elif opt in ('-s', '--sensibilidad'):
        sensibilidad = True
    elif opt in ('-v','--valorar'):
        valorar = True
    elif opt in ('-pE', '--pickleE'):
        rutaPickleE = path.abspath(arg)
        modoPickleE = True
    elif opt in ('-pS', '--pickleS'):
        rutaPickleS = path.abspath(arg)
        modoPickleS = True
    elif opt in ('-pM', '--pickleM'):
        rutaPickleM = path.abspath(arg)
        modoPickleM = True
    elif opt in ('-o', '--salida'):
        rutaSalida = path.abspath(arg)
        print(" rE:rS:e:s:v:pE:pS:pM:o:h'\n['rutaE=', 'rutaS=', 'entrenar', 'sensibilidad', 'valorar', 'pickleE=', 'pickleS=', 'pickleM=', 'salida', 'help'])")
    else:
        print("La variable de linea de comandos no es valida \n'rE:rS:e:s:v:pE:pS:pM:o:h'\n['rutaE=', 'rutaS=', 'entrenar', 'sensibilidad', 'valorar', 'pickleE=', 'pickleS=', 'pickleM=', 'salida', 'help'])")
        print("He fallado en {}".format(opt))
        exit(0)

if entrenar:
    #Comprobar si las variables de entrada estan bien
    if not any((rutaE, rutaS)):
        print("Las variables introducidas no son correctas")
        print("rutaE: {}".format(rutaE))
        print("rutaS: {}".format(rutaS))
        exit(2)
    
    #Comprobar si tengo una variable Pickle de entrenamiento
    if modoPickleE == False:    
        variables = None
        salidas = None
        listaArchivos = listdir(rutaE)
        #Por cada archivo leer el contenido del csv
        for archivo in listaArchivos:
            print(archivo)
            #Si no es archivo continuo
            if path.isfile(path.join(rutaE, archivo)) == False:
                continue
       
            datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(rutaE , archivo)), inCsv)
            #Comprobar longitud de los archivos
            if 'ATQ' in archivo:
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
            else:
                    numeroVentanas = 1
                    longitudDatos = 400
                    datos = datos[0 : longitudDatos]
            
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
            respuestaUsuario = input("Desea guardar los resultados de las variables/salidas? (y/n)")
            if (respuestaUsuario == 'y' or respuestaUsuario == 'n'):
                break
        if respuestaUsuario == 'y':
           if rutaPickleE is None:
               rutaPickleE = path.abspath(path.join(getcwd(),"variablesEntrenamiento.pickle"))
           diccionario = {'VARIABLES' : variables, 'SALIDAS' : salidas}        
           pickle.dump(diccionario, open(rutaPickleE, "wb" ))
           print("Variables guardadas en el archivo: {}".format(rutaPickleE))

    elif modoPickleE == True:
        if path.isfile(rutaPickleE) == False:
            print("rutaPickleE no esta definido o no es archivo")
            exit(2)

        diccionario = pickle.load(open(rutaPickleE, 'rb'))
        variables = diccionario['VARIABLES']
        salidas = diccionario['SALIDAS']
    
    #Aqui ya tengo las variables en variables y salidas ya sea por pickle o analisis
    print ("------------------------------------------------------")
    print ("Ya he leido todos los datos, con las condiciones impuestas")
    print (" * Numero de ataques: {}".format(sum(x for x in salidas if x==1)))
    print (" * Numero de movimientos: {}".format(salidas.shape[0] - sum(x for x in salidas if x==1)))
    _ = input("Pulse una tecla para entrenar...")

    #ENTRENAMOS UN MODELO SVM con CV con gridsearch
    
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    #Haremos 4 SVM uno por cada ataque
    variableCombinada = np.concatenate((variables, salidas.reshape(len(salidas),1)), axis=1)
    modelos = dict()

    for k in range(1, 5):
        variableCombinadaModeloK = variableCombinada[ variableCombinada[:, 12] == k ]
        #Elimino la columna de la posicion
        entradaModeloK = variableCombinadaModeloK[:, :-2]
        print("Entrenar Ubicacion: {}".format(funciones.DevolverUbicacion(k)))
        print("Shape EntradaModeloK: {}".format(entradaModeloK.shape))
        salidaModeloK = variableCombinadaModeloK[:, -1]
        print("Shape SalidaModeloK: {}".format(salidaModeloK.shape))
        print("Numero de ataques: {}".format(sum(salidaModeloK[salidaModeloK == 1])))
        scalerK = StandardScaler().fit(entradaModeloK)
        entradaModeloK = scalerK.transform(entradaModeloK)
        
        #Fusionar ambas
        datos = np.concatenate(entradaModeloK, salidadModeloK)
        dnn = DNN.DeepNeuralNetwork(12, 128, 128, 2)


        parametersK = {'kernel':('rbf', 'sigmoid', 'poly'), 'C':[0.01, 0.1, 1, 10, 100, 1000, 10000]}
        svcK = svm.SVC()
        clfK = GridSearchCV(svcK, parametersK)
        clfK.fit(entradaModeloK, salidaModeloK)
        modelos["SCALER" + str(k)] = scalerK
        modelos["SVC" + str(k)] = clfK.best_estimator_

    print ("------------------------------------------------------")
    print ("Ya he entrenado {} modelos".format(k))
    _ = input("Pulse una tecla para evaluar...")

    #Una vez entranado calculamos como saldría el resultado con el dataset de sensibilidad
    if modoPickleS == False:
        listaArchivosSensibilidad = listdir(rutaS)
        variables = None
        salidas = None
        for archivo in listaArchivosSensibilidad:
            print(archivo)
            #Si no es archivo continuo
            if path.isfile(path.join(rutaS, archivo)) == False:
                continue
           
            datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(rutaS , archivo)), inCsv)
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
               rutaPickleS = path.abspath(path.join(getcwd(),"variablesSensibilidad.pickle"))
           diccionario = {'VARIABLES' : variables, 'SALIDAS' : salidas}        
           pickle.dump(diccionario, open(rutaPickleS, "wb" ))
           print("Variables guardadas en el archivo: {}".format(rutaPickleS))

    elif modoPickleS == True:
        if path.isfile(rutaPickleS) == False:
            print("rutaPickle no esta definido o no es un archivo")
            exit(2)

        diccionario = pickle.load(open(rutaPickleS, 'rb'))
        variables = diccionario['VARIABLES']
        salidas = diccionario['SALIDAS']
    
    print ("------------------------------------------------------")
    print ("Ya he leido todos los datos")
    print (" * Numero de ataques: {}".format(sum(x for x in salidas if x==1)))
    print (" * Numero de movimientos: {}".format(salidas.shape[0] - sum(x for x in salidas if x==1)))
    _ = input("Pulse una tecla para predecir...")

    from sklearn.metrics import accuracy_score
    variableCombinada = np.concatenate((variables, salidas.reshape(len(salidas),1)), axis=1)
    for k in range(1, 5):
        print("Prediccion con modelo {}".format(funciones.DevolverUbicacion(k)))
        variableCombinadaModeloK = variableCombinada[ variableCombinada[:, 12] == k ]
        entradaModeloK = variableCombinadaModeloK[:, :-2]
        print("Shape entradaModelo: {}".format(entradaModeloK.shape))
        salidaModeloK = variableCombinadaModeloK[:, -1]
        print("Shape salidaModelo: {}".format(salidaModeloK.shape))
        scalerK = modelos["SCALER" + str(k)]
        scalerK.transform(entradaModeloK[:,:])
        clfK = modelos["SVC" + str(k)]
        prediccionK = clfK.predict(entradaModeloK)
        print("-------------------------------------------------")
        print("RESULTADO DEL ESTIMADOR {}: {}".format(k, accuracy_score(salidaModeloK, prediccionK)))
        
    while True:
        respuestaUsuario = input("Desea guardar los estimadores? (y/n)")
        if (respuestaUsuario == 'y' or respuestaUsuario == 'n'):
            break
    if respuestaUsuario == 'y':
        if rutaPickleM is None:
            rutaPickleM = path.abspath(input("Ruta del modelo"))
        pickle.dump(modelos, open(rutaPickleM, "wb" ))
        print("Estimador guardado en el archivo: {}".format(rutaPickleM))

    
if sensibilidad:
    #Comprobar que las cosas esten bien
    if not any((rutaS, rutaPickleM)):
        print("Las variables introducidas no son correctas")
        print("rutaS: {}".format(rutaS))
        print("rutaPickleM: {}".format(rutaPickleM))
        exit(2)
    
        
    listaArchivos = listdir(rutaS) #Cargar archivos
    modelos =  pickle.load(open(rutaPickleM, 'rb'))     #Cargar modelos
    var_sensibilidad = None
    sal_sensibilidad = None
    diccionario_archivo = {}
    idx_archivo = 0
    for idx_archivo, archivo in enumerate(listaArchivos):
        print(archivo)
        #SI NO ES ARCHIVO SALTO
        if path.isfile(path.join(rutaS, archivo)) == False:
            continue
        datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(rutaS , archivo)), inCsv)
        #Comprobar longitud de los archivos
        if longitudDatos < 15 * freq:
            continue
        elif longitudDatos >= 15 * freq:
            numeroVentanas = 6
            datos = datos[0 : (15 * freq)] #ojo que es [ : )
            longitudDatos = int(15 * freq)
        diccionario_archivo[idx_archivo] = archivo #guardo en el diccionario el nombre   
        #calcular datos de inicio y fin de las ventanas
        ventanas = funciones.calcularventana(numeroVentanas, freq)
        #calcular ventana hamming para filtrado de extremos
        hammingWindow = funciones.hamming(longitudDatos, freq)
        #rehacer datos con la ventana hamming
        datos = datos * hammingWindow
        #inicializo la ventana a ceros
        variableLocal = np.zeros((numeroVentanas, 12))
        paciente = funciones.devolverpaciente(archivo)
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
        #Seleccion del modelo SVM entrenado
        scaler_sensibilidad = modelos["SCALER" + str(paciente)]
        clf_sensibilidad = modelos["SVC" + str(paciente)]
        scaler_sensibilidad.transform(variableLocal[:,:])
        prediccion_sensibilidad = clf_sensibilidad.predict(variableLocal)
        salida_sensibilidad = (0, 1)['ATQ' in archivo]
        #Montar la variable de salida
        if var_sensibilidad is None:
            var_sensibilidad = np.concatenate(([idx_archivo], [paciente], prediccion_sensibilidad, [salida_sensibilidad]), axis=0)
            print(var_sensibilidad)
        else:
            var_sensibilidad = np.vstack((var_sensibilidad, np.concatenate(([idx_archivo], [paciente], prediccion_sensibilidad, [salida_sensibilidad]), axis=0)))
            print(var_sensibilidad[-1,:])
        
    
    print("____________________________")
    print("Shape de variable var_sensibilidad: {}".format(var_sensibilidad.shape))
    print("Numero de archivos ataques {}".format(np.sum(var_sensibilidad[:,-1])))
    #Ya tenemos montado la variable var_sensibilidad
    for paciente in range(1, num_pacientes+1):
        print("Iteracion para paciente {}".format(funciones.DevolverUbicacion(paciente)))
        var_usar = var_sensibilidad[var_sensibilidad[:,1] == paciente]
        print("   Shape de la matriz a usar {}".format(var_usar.shape))
        print("   Numero de ataques {}".format(np.sum(var_usar[:,-1])))
        _ = input("Pulsa")
        resultado_maximo = 0
        for minima_sensibilidad in range(1,7):
            pred = np.sum(var_usar[:,2:8], axis=1) >= minima_sensibilidad #es array logico
            pred = pred * 1 #es 1 y 0
            resultado = funciones.CalcularResultado(pred, var_usar[:,-1])
            if (resultado > resultado_maximo):
                resultado_maximo = resultado
                print("    El resultado con Sensibilidad Minima {} es {}  *BEST".format(minima_sensibilidad, resultado))
            else:
                print("    El resultado con Sensibilidad Minima {} es {}".format(minima_sensibilidad, resultado))
         


    
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
