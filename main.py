# Cargar las librerías
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qsar_analysis import chembl_screening as cs
from qsar_analysis import qsar_modelling as qm

# Carga y cribado de los datos       
## Cargar los dataframes
data_raw = pd.read_csv(sys.argv[1], sep=';')
data_std = pd.read_csv(sys.argv[2], sep='\t', index_col=[0])

## Inner Join y eliminación de NAs en el valor de pChEMBL
data = pd.merge(data_std, data_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data = data[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey", 
                                 "properties_log", "Molecular Weight",
                                 "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
                                 
## Eliminación de las moléculas incorrectas
data = cs.molPropCorrect(data)

## Eliminar duplicados
tupl = cs.delDupMols(data)
if tupl != 0: data_max, data_mean = tupl

## Obtener los datos bidimensionales para el DBSCAN
dbscan_max = data_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_mean = data_mean[["Molecular Weight", "pChEMBL Value"]]

## Elbow Plot
n_neighbors = 15
knee_max = cs.elbowPlot(dbscan_max, n_neighbors)
print("Codo para la versión de máximos: ", knee_max)
knee_mean = cs.elbowPlot(dbscan_mean, n_neighbors)
print("Codo para la versión de medias: ", knee_mean)
plt.show()

## Histogramas de la distribución de pChEMBL
### Max
plt.hist(x=np.asarray(dbscan_max["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL (Max)')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### Mean
plt.hist(x=np.asarray(dbscan_mean["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL (Mean)')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

## Análisis DBSCAN
cs.dbscanPlot(dbscan_max, knee_max, n_neighbors, 'DBSCAN (Max)')
cs.dbscanPlot(dbscan_mean, knee_mean, n_neighbors, 'DBSCAN (Mean)')

## Categorización de pChEMBL
data_max = cs.labelData(data_max)
data_mean = cs.labelData(data_mean)

## Eliminar las columnas sobrantes
cs.dropCols(data_max)
cs.dropCols(data_mean)


# Generación de los modelos QSAR

## Obtener los fingerprints de los códigos SMILE
### Max
mols_max = qm.getMols(data_max[["Molecule ChEMBL ID", "standardized_molecule"]])
fingerprints_max = qm.getMorganFP(mols_max)
data_max = pd.merge(fingerprints_max, data_max, on='Molecule ChEMBL ID').drop(["standardized_molecule"], axis=1)

### Mean
mols_mean = qm.getMols(data_mean[["Molecule ChEMBL ID", "standardized_molecule"]])
fingerprints_mean = qm.getMorganFP(mols_mean)
data_mean = pd.merge(fingerprints_mean, data_mean, on='Molecule ChEMBL ID').drop(["standardized_molecule"], axis=1)

## Dividir el conjunto de datos en train y test
data_train_max, data_test_max = qm.subsetData(data_max, 0.8, 97)
data_train_mean, data_test_mean = qm.subsetData(data_mean, 0.8, 97)
print(len(data_test_max))

