# -*- coding: utf-8 -*-
"""
@author: Diego
"""
import numpy as np
import pandas as pd
import scipy.stats as st

def hamming(longitud: int, frecuencia: int):
    if longitud < (0.2 * int(frecuencia)):
        return (np.ones(longitud))
    else:
        i = np.linspace(0, 1, int(0.1 * frecuencia))
        i = np.concatenate((i, np.ones(int(longitud - (0.2 * frecuencia)))), axis=0)
        i = np.concatenate((i, np.linspace(1, 0, int(0.1 * frecuencia))), axis=0)
        return(i)
        
def calcularventana (numeroventana: int, frecuencia: int):
    ventana = np.zeros((int(numeroventana),2))
    for i in range(numeroventana):
        ventana[i , 0] = 2.5 * frecuencia * i #inicio
        ventana[i , 1] = (2.5 * frecuencia * (i + 1)) - 1   #fin
    return(ventana)
    
def leerarchivocsv (archivo, columna):
    df = pd.read_csv(archivo, float_precision='high')
    a = df.iloc[: , columna].values
    return (a, a.shape[0])

def calcularenergia(datos):
    media = datos.mean()
    energia = (1/400) * np.sum(datos - media) #Energia por segundo
    return (energia)

def calcularRMS(datos):
    rms = np.sqrt(np.sum(datos * datos) / datos.shape[0])
    return (rms)

def calcularenergiaespectral(datos):
    return ()

def calcularVpp(datos):
    return (np.max(datos) - np.min(datos))

def autocorr(x):
    norm = x - np.mean(x)
    result = np.correlate(norm, norm, mode='full')
    acorr = result[int(result.size/2):]
    acorr /= ( x.var() * np.arange(x.size, 0, -1) )
    return np.sum(acorr)
        
def calcularfftyee(datos, frecuencia):
    fourierComponents = np.fft.fft(datos)
    fourierCoefficients = np.abs(fourierComponents)
    fourierFrequencies = np.array(range(fourierComponents.size))
    fourierFrequencies = fourierFrequencies * (frecuencia / fourierComponents.size)

    fft0to8 = np.sum(fourierCoefficients[(fourierFrequencies >=0.4) & (fourierFrequencies <=8)])
    fft8to16 = np.sum(fourierCoefficients[(fourierFrequencies > 8) & (fourierFrequencies <= 16)])
    fft16to30 = np.sum(fourierCoefficients[(fourierFrequencies > 16) & (fourierFrequencies <= 30)])
    ee = np.sqrt(np.sum((fourierCoefficients * fourierFrequencies)**2))
    return (fft0to8, fft8to16, fft16to30, ee)

def calcularestadisticos(datos):
    #8-> kur 9-> skew , 10 -> var, 11 -> entropia
    return (st.kurtosis(datos), st.skew(datos), datos.var())

def calcularentropia(datos):
    return(st.entropy(datos))

def devolverpaciente(archivo):
    if 'BCN' in archivo:
        return (1)
    elif 'ZGZ' in archivo:
        return (2)
    elif 'POZ' in archivo:
        return (3)
    else:
        return (4) #GET

def DevolverUbicacion(numero):
    dic = {1:"BCN" , 2:"ZGZ", 3:"POZ", 4:"GET"}
    return (dic[numero])

def CalcularResultado(pred, real):
    n = pred[(pred==1) & (real==1)].shape[0]
    m = pred[(pred==0) & (real==0)].shape[0]
    N = real[real==1].shape[0]
    M = real.shape[0] - N
    return(55 * (n / N) + 42.5 * (m / M))

def leer_csv_2in (archivo, columna1, columna2):
    df = pd.read_csv(archivo, float_precision='high')
    a = df.iloc[: , columna1].values
    b = df.iloc[: , columna2].values
    return (a,b, a.shape[0])


