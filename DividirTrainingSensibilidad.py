import os
from sklearn.model_selection import train_test_split
import pdb
import shutil


patho = "/home/diego/Documents/DataChallenge/training/"
listaArchivos = os.listdir(os.path.abspath(patho))

listaDefinitiva = []
salidaDefinitiva = []
for arch in listaArchivos:	
	if os.path.isfile(os.path.join(patho, arch)) == False:
		continue
	
	
	listaDefinitiva.append(arch)
	if 'ATQ' in arch:
		salidaDefinitiva.append(int(1))
	else: 
		salidaDefinitiva.append(int(0))


X_train, X_sensibilidad, y_train, y_test = train_test_split(listaDefinitiva,salidaDefinitiva, test_size=0.3, random_state=42, shuffle=True)
print(X_train)
print("\n")
print(X_sensibilidad)
print("\n")
print(len(listaDefinitiva))

print("\n")
print(len(salidaDefinitiva))

for arch in X_sensibilidad:
    shutil.move(os.path.join(patho,arch), os.path.abspath( "/home/diego/Documents/DataChallenge/sensibilidad/" + arch))


