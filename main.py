# Cargar las librerías
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from qsar_pckg import chembl_screening as cs
from qsar_pckg import qsar_modelling as qm

# Carga y cribado de los datos
## Carga de los datos   
### DNMT1
data_raw = pd.read_csv(sys.argv[1], sep=';')
data_std = pd.read_csv(sys.argv[2], sep='\t', index_col=[0])

### DNMT3A
data_dnmt3a_raw = pd.read_csv(sys.argv[3], sep=';')
data_dnmt3a_std = pd.read_csv(sys.argv[4], sep='\t', index_col=[0])

### DNMT3B
data_dnmt3b_raw = pd.read_csv(sys.argv[5], sep=';')
data_dnmt3b_std = pd.read_csv(sys.argv[6], sep='\t', index_col=[0])

### Oleacina
oleacin = pd.read_csv(sys.argv[7], sep='\t', index_col=[0])

## Inner Join y eliminación de NAs en el valor de pChEMBL
### DNMT1
data = pd.merge(data_std, data_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data = data[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey", 
             "properties_log", "Molecular Weight", "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
                                 
### DNMT3A
data_dnmt3a = pd.merge(data_dnmt3a_std, data_dnmt3a_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data_dnmt3a = data_dnmt3a[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey", 
                           "properties_log", "Molecular Weight", "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
                                 
### DNMT3B
data_dnmt3b = pd.merge(data_dnmt3b_std, data_dnmt3b_raw,left_on='compound',right_on='Molecule ChEMBL ID')
data_dnmt3b = data_dnmt3b[["Molecule ChEMBL ID", "standardized_molecule", "standardized_molecule_InchiKey", 
                           "properties_log", "Molecular Weight", "pChEMBL Value"]].dropna(subset = ["pChEMBL Value"])
                                 
## Eliminación de las moléculas incorrectas
data = cs.molPropCorrect(data)
data_dnmt3a = cs.molPropCorrect(data_dnmt3a)
data_dnmt3b = cs.molPropCorrect(data_dnmt3b)

## Eliminar duplicados
uniq = cs.delDupMols(data)
if uniq != 0: data_max, data_mean = uniq
uniq = cs.delDupMols(data_dnmt3a)
if uniq != 0: data_dnmt3a_max, data_dnmt3a_mean = uniq
uniq = cs.delDupMols(data_dnmt3b)
if uniq != 0: data_dnmt3b_max, data_dnmt3b_mean = uniq

## Obtener los datos bidimensionales para el DBSCAN
dbscan_max = data_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_mean = data_mean[["Molecular Weight", "pChEMBL Value"]]
dbscan_dnmt3a_max = data_dnmt3a_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_dnmt3a_mean = data_dnmt3a_mean[["Molecular Weight", "pChEMBL Value"]]
dbscan_dnmt3b_max = data_dnmt3b_max[["Molecular Weight", "pChEMBL Value"]]
dbscan_dnmt3b_mean = data_dnmt3b_mean[["Molecular Weight", "pChEMBL Value"]]

## Elbow Plot
n_neighbors = 15

### DNMT1
knee_max = cs.elbowPlot(dbscan_max, n_neighbors)
print("Codo para la versión de máximos: ", knee_max)
knee_mean = cs.elbowPlot(dbscan_mean, n_neighbors)
print("Codo para la versión de medias: ", knee_mean)

### DNMT3A
knee_dnmt3a_max = cs.elbowPlot(dbscan_dnmt3a_max, n_neighbors)
print("Codo para la versión de máximos: ", knee_dnmt3a_max)
knee_dnmt3a_mean = cs.elbowPlot(dbscan_dnmt3a_mean, n_neighbors)
print("Codo para la versión de medias: ", knee_dnmt3a_mean)

### DNMT3B
knee_dnmt3b_max = cs.elbowPlot(dbscan_dnmt3b_max, n_neighbors)
print("Codo para la versión de máximos: ", knee_dnmt3b_max)
knee_dnmt3b_mean = cs.elbowPlot(dbscan_dnmt3b_mean, n_neighbors)
print("Codo para la versión de medias: ", knee_dnmt3b_mean)
plt.show()

## Histogramas de la distribución de pChEMBL
### DNMT1
#### Max
plt.hist(x=np.asarray(dbscan_max["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL (Max) para DNMT1')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

#### Mean
plt.hist(x=np.asarray(dbscan_mean["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL (Mean) para DNMT1')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### DNMT3A
#### Max
plt.hist(x=np.asarray(dbscan_dnmt3a_max["pChEMBL Value"], dtype='float'), bins=15, color='#71E3D9', rwidth=0.85)
plt.title('Histograma del pChEMBL (Max) para DNMT3A')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

#### Mean
plt.hist(x=np.asarray(dbscan_dnmt3a_mean["pChEMBL Value"], dtype='float'), bins=15, color='#71E3D9', rwidth=0.85)
plt.title('Histograma del pChEMBL (Mean) para DNMT3A')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### DNMT3B
#### Max
plt.hist(x=np.asarray(dbscan_dnmt3b_max["pChEMBL Value"], dtype='float'), bins=15, color='#A775F5', rwidth=0.85)
plt.title('Histograma del pChEMBL (Max) para DNMT3B')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

#### Mean
plt.hist(x=np.asarray(dbscan_dnmt3b_mean["pChEMBL Value"], dtype='float'), bins=15, color='#A775F5', rwidth=0.85)
plt.title('Histograma del pChEMBL (Mean) para DNMT3B')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

## Análisis DBSCAN
cs.dbscanPlot(dbscan_max, knee_max, n_neighbors, 'DBSCAN DNMT1 (Max)')
cs.dbscanPlot(dbscan_mean, knee_mean, n_neighbors, 'DBSCAN DNMT1 (Mean)')
cs.dbscanPlot(dbscan_dnmt3a_max, knee_dnmt3a_max, n_neighbors, 'DBSCAN DNMT3A (Max)')
cs.dbscanPlot(dbscan_dnmt3a_mean, knee_dnmt3a_mean, n_neighbors, 'DBSCAN DNMT3A (Mean)')
cs.dbscanPlot(dbscan_dnmt3b_max, knee_dnmt3b_max, n_neighbors, 'DBSCAN DNMT3B (Max)')
cs.dbscanPlot(dbscan_dnmt3b_mean, knee_dnmt3b_mean, n_neighbors, 'DBSCAN DNMT3B (Mean)')

## Categorización de pChEMBL
### Dataframes DNMTX
data_max = cs.labelData(data_max)
data_mean = cs.labelData(data_mean)
data_dnmt3a_max = cs.labelData(data_dnmt3a_max)
data_dnmt3a_mean = cs.labelData(data_dnmt3a_mean)
data_dnmt3b_max = cs.labelData(data_dnmt3b_max)
data_dnmt3b_mean = cs.labelData(data_dnmt3b_mean)

### Oleacina
oleacin.drop(["original_molecule", "standardized_molecule_InchiKey", "properties_log"], axis=1, inplace=True)
oleacin['Activity'] = 'Active'
oleacin['Activity'] = pd.Categorical(oleacin['Activity'], categories=["Inactive", "Active"])

## Eliminar las columnas sobrantes
cs.dropCols(data_max)
cs.dropCols(data_mean)
cs.dropCols(data_dnmt3a_max)
cs.dropCols(data_dnmt3a_mean)
cs.dropCols(data_dnmt3b_max)
cs.dropCols(data_dnmt3b_mean)

# Crear los datasets para cada caso de estudio

## DNMT3
### Obtener las moléculas activas de DNMT3X
dnmt3_max = data_dnmt3a_max.append(data_dnmt3b_max, ignore_index=True)
dnmt3_mean = data_dnmt3a_mean.append(data_dnmt3b_mean, ignore_index=True)
dnmt3_actives_max = dnmt3_max.loc[dnmt3_max['Activity'] == 'Active']
dnmt3_actives_mean = dnmt3_mean.loc[dnmt3_mean['Activity'] == 'Active']

### Obtener el desbalanceo de clases cuantitativamente
n_actives_max = data_max.loc[data_max['Activity'] == 'Active'].shape[0]
n_actives_mean = data_mean.loc[data_mean['Activity'] == 'Active'].shape[0]
n_inactives_max = data_max.shape[0] - n_actives_max
n_inactives_mean = data_mean.shape[0] - n_actives_mean

n_imbalance_max = n_inactives_max - n_actives_max
n_imbalance_mean = n_inactives_mean - n_actives_mean

### Balancear los sets originales
if (n_imbalance_max >= dnmt3_actives_max.shape[0]): data3_max = data_max.append(dnmt3_actives_max)
else: data3_max = data_max.append(dnmt3_actives_max.sample(n = n_imbalance_max))
if (n_imbalance_mean >= dnmt3_actives_mean.shape[0]): data3_mean = data_mean.append(dnmt3_actives_mean)
else: data3_mean = data_mean.append(dnmt3_actives_mean.sample(n = n_imbalance_mean))

## Oleacin
dataO_max = data_max.append(oleacin)
dataO_mean = data_mean.append(oleacin)

# Preparación de los conjuntos de entrenamiento

## Dividir los conjuntos de datos en train y test
### DNMT1
train_max_X, test_max_X, train_max_Y, test_max_Y, max_X, max_Y = qm.splitData(data_max, 'train_max_226')
train_mean_X, test_mean_X, train_mean_Y, test_mean_Y, mean_X, mean_Y = qm.splitData(data_mean, 'train_mean_226')

### DNMT3
train3_max_X, test3_max_X, train3_max_Y, test3_max_Y, max3_X, max3_Y = qm.splitData(data3_max, 'train3_max_226')
train3_mean_X, test3_mean_X, train3_mean_Y, test3_mean_Y, mean3_X, mean3_Y = qm.splitData(data3_mean, 'train3_mean_226')

### Oleacin
trainO_max_X, testO_max_X, trainO_max_Y, testO_max_Y, maxO_X, maxO_Y = qm.splitData(dataO_max, 'trainO_max_226')
trainO_mean_X, testO_mean_X, trainO_mean_Y, testO_mean_Y, meanO_X, meanO_Y = qm.splitData(dataO_mean, 'trainO_mean_226')

## Obtener los fingerprints de los códigos SMILE
### DNMT1
train_max_X, test_max_X = qm.getTrainTestFP(train_max_X, test_max_X, max_X)
train_mean_X, test_mean_X = qm.getTrainTestFP(train_mean_X, test_mean_X, mean_X)

### DNMT3
train3_max_X, test3_max_X = qm.getTrainTestFP(train3_max_X, test3_max_X, max3_X)
train3_mean_X, test3_mean_X = qm.getTrainTestFP(train3_mean_X, test3_mean_X, mean3_X)

### Oleacin
trainO_max_X, testO_max_X = qm.getTrainTestFP(trainO_max_X, testO_max_X, maxO_X)
trainO_mean_X, testO_mean_X = qm.getTrainTestFP(trainO_mean_X, testO_mean_X, meanO_X)

# Generación de los modelos QSAR

## Definir el método de crosvalidación, en este caso 10 pliegues estratificados
kf = StratifiedKFold(10)

## Definir los parámetros para el tunning de los modelos
### Random Forest
params_rf = {'bootstrap': [True, False],
                 'max_depth': [5, 7, 10, None],
                 'n_estimators': [50, 100, 200, 300, 500]}

### Support Vector Machines
params_svm = {'C': [0.1, 0.5, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

### Naive Bayes
params_nb = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}

### XGradientBoostTree
params_xgb = {'min_child_weight': [1, 5, 10],
                  'gamma': [0.5, 1, 1.5, 2, 5],
                  'max_depth': [3, 4, 5]}

### Artificial Neural Network
params_ann = {'batch_size': [10, 20, 50],
                  'epochs': [50, 100],
                  'n': [10, 50, 100]}

## Realizar el entrenamiento de los modelos QSAR
qm.QSARtraining(train_max_X, train_max_Y, train_mean_X, train_mean_Y, kf, params_rf, params_svm, params_nb, params_xgb, params_ann, "_DNMT1")
qm.QSARtraining(train3_max_X, train3_max_Y, train3_mean_X, train3_mean_Y, kf, params_rf, params_svm, params_nb, params_xgb, params_ann, "_DNMT3")
qm.QSARtraining(trainO_max_X, trainO_max_Y, trainO_mean_X, trainO_mean_Y, kf, params_rf, params_svm, params_nb, params_xgb, params_ann, "_Oleacin")

## Almacenar los conjuntos de test
qm.saveTestData(test_max_X, test_max_Y, test_mean_X, test_mean_Y, "_DNMT1")
qm.saveTestData(test3_max_X, test3_max_Y, test3_mean_X, test3_mean_Y, "_DNMT3")
qm.saveTestData(testO_max_X, testO_max_Y, testO_mean_X, testO_mean_Y, "_Oleacin")








