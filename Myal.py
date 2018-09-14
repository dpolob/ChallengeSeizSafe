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



#inVar = np.array([0,1,3,4,5,7])


print("Frecuencia: {}\n".format(freq))
print("CWD       : {}\n".format(cwd))

options, remainder = getopt.getopt(sys.argv[1:], 'e:s:v:', ['version', 'entrenar=', 'sensibilidad=','valorar=','modelo='])
training, sensibilidad, valorar = (False, False, False) 

#Leer linea de comandos del programa
for opt, arg in options:
    if opt in ('-e', '--entrenar'):
        training = True
        normalizedPath = path.abspath(arg)
    elif opt in ('-s', '--sensibilidad'):
        sensibilidad = True
        normalizedPath = path.abspath(arg)
    elif opt in ('-v','--valorar'):
        valorar = True
        archivoSalida = path.abspath(arg)
    elif opt in ('-m','--modelo'):
        rutaModelo = path.abspath(arg)
    elif opt == '--version':
        print(version)
        exit(0)
    else:
       exit(2)



if training:    
    variables = None
    salidas = None

    listaArchivos = listdir(normalizedPath)
    
    #Por cada archivo leer el contenido del csv
    for archivo in listaArchivos:
        print(archivo + "\n")
        
        #SI NO ES ARCHIVO SALTO
        if path.isfile(path.join(normalizedPath, archivo)) == False:
            continue
    
      
        datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(normalizedPath , archivo)), inCsv)
        
        #Comprobar longitud de los archivos
        if 'ATQ' in archivo: #Si es un ataque entonces se procesa entero
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
        elif 'MOV' in archivo: # Si es un movimiento solo se procesa una ventana si 6000<long<24000
            if (longitudDatos < 24000 & longitudDatos > 6000):
                datos = datos[0 : 2499]
            else:
                continue


        #calcular datos de inicio y fin de las ventanas
        ventanas = funciones.calcularventana(numeroVentanas, freq)
        
        #calcular ventana hamming para filtrado de extremos
        hammingWindow = funciones.hamming(longitudDatos, freq)
        
        #rehacer datos con la ventana hamming
        datos = datos * hammingWindow

        #inicializo la ventana a ceros
        variableLocal = np.zeros((numeroVentanas, 12))
        
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
        
        pdb.set_trace()
        if variables is None: #es la primera vez y variables y salida debe ser inicializado
            variables = variableLocal
            salidas = salidaLocal
        else:
            variables = np.vstack((variables, variableLocal))
            salidas = np.append(salidas, salidaLocal)
    
        #Aqui ya tengo las variables en variables y salidas
        print ("------------------------------------------------------\n")
        print ("Ya he leido todos los datos, con las condiciones impuestas\n")
        print (" * Numero de ataques: {}\n".format(sum(x for x in salida if x==1)))
        print (" * Numero de movimientos: {}\n".format(salida.shape[0] - sum(x for x in salida if x==1)))
        tirar = input("Pulse una tecla para entrenar...\n")

        
        #ENTRENAMOS UN MODELO SVM con CV con gridsearch
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(variables[:,:], salidas, test_size=0.50, random_state=42, shuffle=True)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(variables[:,:])
        #X_train = scaler.transform(X_train)
        #X_test = scaler.transform(X_test)
        
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        parameters = {'kernel':('rbf', 'sigmoid', 'poly'), 'C':[0.01, 0.1, 1, 10, 100, 1000]}
        svc = svm.SVC(verbose=True)
        clf = GridSearchCV(svc, parameters, verbose=True)
        clf.fit(variables, salidas)
        
        prediccion = clf.predict(X_test)
        from sklearn.metrics import accuracy_score
        print("-------------------------------------------------\n")
        print("RESULTADO DEL ESTIMADOR: {}\n".format(accuracy_score(y_test, prediccion)))
        
        bestModel = {'SVM': clf.best_estimator_ , 'SCALER': scaler}
        
        while True:
            respuestaUsuario = input("Desea guardar el estimador? (y/n)\n")
            if (respuestaUsuario == 'y' | respuestaUsuario == 'n'):
                break

        if respuestaUsuario == 'y':
            #Serializar
            pickle.dump(bestModel, open(rutaModelo, "wb" ))
            print("Estimador guardado en el archivo: {}\n".format(rutaModelo))
               
    
if sensibilidad:
    listaArchivos = listdir(normalizedPath)
    variables = None
    salidas = None

    #Por cada archivo leer el contenido del csv
    for archivo in listaArchivos:
        print(archivo + "\n")
        
        #SI NO ES ARCHIVO SALTO
        if path.isfile(path.join(normalizedPath, archivo)) == False:
            continue
          
        datos, longitudDatos = funciones.leerarchivocsv(path.abspath(path.join(normalizedPath , archivo)), inCsv)
        
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
     
    #Cargar estimador y scaler
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    import pickle

    pickle.load(diccionario, open(rutaModelo, 'rb'))
    modelo = diccionario['SVM']
    scaler = diccionario['SCALER']


    modelo, scaler = # bestModel = {'SVM': clf.best_estimator_ , 'SCALER': scaler

   # svm, scaler = pickle.read(open(rutaModelo, "rb").read()
   #         pickle.dump(bestModel, open(rutaModelo, "wb" ))
   #         pickle.dump(bestModel, open(rutaModelo, "wb" ))
   # scaler = StandardScaler().fit(variables[:,:])
   #Para 
 




        





   

elif valorar:
    print()
    
    


    
