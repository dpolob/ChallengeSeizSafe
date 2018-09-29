from os import path, listdir
import getopt
import pandas as pd
import numpy as np

freq = 400

options, remainder = getopt.getopt(sys.argv[1:], 'o:d:', ['origen=', 'destino='])

for opt, arg in options:
    if opt in ('-o', '--origen'):
        directorio_origen = path.abspath(arg)
    elif opt in ('-d', '--destino'):
        directorio_destino = path.abspath(arg)
    else:
        print("Especifique directorio con la opcion --directorio=<directorio>")
        exit(2)

lista_archivos = listdir(directorio_origen)
i = 1
for archivo in lista_archivos:
    if path.isfile(path.join(directorio_origen, archivo)) == False:
        continue
    
    # Lectura de datos
    df = pd.read_csv(path.join(directorio_origen, archivo), floar_precision='high')
    datos = df.iloc[:,:].values
    
    # Identificacion del paciente
    if 'POZ' in archivo:
        paciente = 'POZ'
    elif 'GET' in archivo:
        paciente = 'GET'
    elif 'BCN' in archivo:
        paciente = 'BCN'
    elif 'ZGZ' in archivo:
        paciente = 'ZGZ'
    
    longitud_datos = datos.shape[0]

    if longitud_datos == (15 * freq):
        EscribirArchivo(datos, path.join(directorio_destino, paciente + str(i)))
    elif longitud_datos < (15 * freq):
        repetir = int(((15 * freq) // longitud_datos) + 1)
        for j in range(repetir):
            datos

    elif longitud_datos > (15 * freq):
        datos = datos[:,(15 * freq)]
        EscribirArchivo(datos, path.join(directorio_destino, paciente + str(i)))
    i= i + 1


def EscribirArchivo(datos, ruta):
    print()



