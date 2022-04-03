# Cargar las librerías
from qsar_analysis import chembl_screening as cs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los dataframes
## DNMT RAW
data_dnmt1_226_raw = pd.read_csv('../Datos/data_dnmt1_226.csv', sep=';')
data_dnmt1_430_raw = pd.read_csv('../Datos/data_dnmt1_430.csv', sep=';')

## DNMT STD
data_dnmt1_226_std = pd.read_csv('../Datos/data_dnmt1_226_std.csv', sep='\t', index_col=[0])
data_dnmt1_430_std = pd.read_csv('../Datos/data_dnmt1_430_std.csv', sep='\t', index_col=[0])

# Inner Join y eliminación de NAs en el valor de pChEMBL
## DNMT1_226
data_dnmt1_226 = pd.merge(data_dnmt1_226_std, data_dnmt1_226_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data_dnmt1_226 = data_dnmt1_226[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey", 
                                 "properties_log", "Molecular Weight",
                                 "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
## DNMT1_430
data_dnmt1_430 = pd.merge(data_dnmt1_430_std, data_dnmt1_430_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data_dnmt1_430 = data_dnmt1_430[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey",
                                 "properties_log", "Molecular Weight",
                                 "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
                                 
# Eliminación de las moléculas incorrectas
data_dnmt1_226 = cs.molPropCorrect(data_dnmt1_226)
data_dnmt1_430 = cs.molPropCorrect(data_dnmt1_430)

# Eliminar duplicados
tupl = cs.delDupMols(data_dnmt1_226)
if tupl != 0: data_dnmt1_226_max, data_dnmt1_226_mean = tupl
tupl = cs.delDupMols(data_dnmt1_430)
if tupl != 0: data_dnmt1_430_max, data_dnmt1_430_mean = tupl

# Obtener los datos bidimensionales para el DBSCAN
## DNMT1_226
dbscan_226_max = data_dnmt1_226_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_226_mean = data_dnmt1_226_mean[["Molecular Weight", "pChEMBL Value"]]
## DNMT1_430
dbscan_430_max = data_dnmt1_430_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_430_mean = data_dnmt1_430_mean[["Molecular Weight", "pChEMBL Value"]]

# Elbow Plot
n_neighbors = 15
knee_226_max = cs.elbowPlot(dbscan_226_max, n_neighbors)
print("Codo para DNMT1_226_max: ", knee_226_max)
knee_226_mean = cs.elbowPlot(dbscan_226_mean, n_neighbors)
print("Codo para DNMT1_226_mean: ", knee_226_mean)
knee_430_max = cs.elbowPlot(dbscan_430_max, n_neighbors)
print("Codo para DNMT1_430_max: ", knee_430_max)
knee_430_mean = cs.elbowPlot(dbscan_430_mean, n_neighbors)
print("Codo para DNMT1_430_mean: ", knee_430_mean)
plt.show()

# Histogramas de la distribución de pChEMBL
## DNMT_226
### Max
plt.hist(x=np.asarray(dbscan_226_max["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL para DNMT1_226_max')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### Mean
plt.hist(x=np.asarray(dbscan_226_mean["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL para DNMT1_226_mean')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

## DNMT_430
### Max
plt.hist(x=np.asarray(dbscan_430_max["pChEMBL Value"], dtype='float'), bins=15, color='#71E3D9', rwidth=0.85)
plt.title('Histograma del pChEMBL para DNMT1_430_max')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### Mean
plt.hist(x=np.asarray(dbscan_430_mean["pChEMBL Value"], dtype='float'), bins=15, color='#71E3D9', rwidth=0.85)
plt.title('Histograma del pChEMBL para DNMT1_430_mean')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

# Análisis DBSCAN
cs.dbscanPlot(dbscan_226_max, knee_226_max, n_neighbors, "DBSCAN DNMT1_226_max")
cs.dbscanPlot(dbscan_226_mean, knee_226_mean, n_neighbors, "DBSCAN DNMT1_226_mean")
cs.dbscanPlot(dbscan_430_max, knee_430_max, n_neighbors, "DBSCAN DNMT1_430_max")
cs.dbscanPlot(dbscan_430_mean, knee_430_mean, n_neighbors, "DBSCAN DNMT1_430_mean")

# Categorización de pChEMBL
data_dnmt1_226_max = cs.labelData(data_dnmt1_226_max)
data_dnmt1_226_mean = cs.labelData(data_dnmt1_226_mean)
data_dnmt1_430_max = cs.labelData(data_dnmt1_430_max)
data_dnmt1_430_mean = cs.labelData(data_dnmt1_430_mean)

# Eliminar las columnas sobrantes
data_dnmt1_226_max = cs.dropCols(data_dnmt1_226_max)
data_dnmt1_226_mean = cs.dropCols(data_dnmt1_226_mean)
data_dnmt1_430_max = cs.dropCols(data_dnmt1_430_max)
data_dnmt1_430_mean = cs.dropCols(data_dnmt1_430_mean)



