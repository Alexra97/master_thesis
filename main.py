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
                           
## Unión DNMT3A-DNMT3B y diferencia con DNMT1
data_dnmt3 = data_dnmt3a.append(data_dnmt3b, ignore_index=True)
uniq_ids = [m for m in data_dnmt3["Molecule ChEMBL ID"].tolist() if m not in data["Molecule ChEMBL ID"].tolist()]
data_dnmt3 = data_dnmt3.loc[data_dnmt3["Molecule ChEMBL ID"].isin(uniq_ids)]
                                 
## Eliminación de las moléculas incorrectas
data = cs.molPropCorrect(data)
data_dnmt3 = cs.molPropCorrect(data_dnmt3)

## Eliminar duplicados
uniq = cs.delDupMols(data)
if uniq != 0: data_max, data_mean = uniq
uniq = cs.delDupMols(data_dnmt3)
if uniq != 0: data_dnmt3_max, data_dnmt3_mean = uniq

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
plt.title('Histograma del pChEMBL (Max) para DNMT1')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

### Mean
plt.hist(x=np.asarray(dbscan_mean["pChEMBL Value"], dtype='float'), bins=15, color='#F2AB6D', rwidth=0.85)
plt.title('Histograma del pChEMBL (Mean) para DNMT1')
plt.xlabel('pChEMBL')
plt.ylabel('Frecuencia')
plt.show()

## Análisis DBSCAN
cs.dbscanPlot(dbscan_max, knee_max, n_neighbors, 'DBSCAN DNMT1 (Max)')
cs.dbscanPlot(dbscan_mean, knee_mean, n_neighbors, 'DBSCAN DNMT1 (Mean)')

## Categorización de pChEMBL
### Dataframes DNMTX
data_max = cs.labelData(data_max)
data_mean = cs.labelData(data_mean)
data_dnmt3_max = cs.labelData(data_dnmt3_max)
data_dnmt3_mean = cs.labelData(data_dnmt3_mean)

### Oleacina
oleacin.rename(columns={'compound': 'Molecule ChEMBL ID'}, inplace=True)
oleacin.drop(["original_molecule", "standardized_molecule_InchiKey", "properties_log"], axis=1, inplace=True)
oleacin['Activity'] = 'Active'
oleacin['Activity'] = pd.Categorical(oleacin['Activity'], categories=["Inactive", "Active"])

## Eliminar las columnas sobrantes
cs.dropCols(data_max)
cs.dropCols(data_mean)
cs.dropCols(data_dnmt3_max)
cs.dropCols(data_dnmt3_mean)

# Crear los datasets para cada caso de estudio

## DNMT3
### Obtener las moléculas activas de DNMT3X
dnmt3_actives_max = data_dnmt3_max.loc[data_dnmt3_max['Activity'] == 'Active']
dnmt3_actives_mean = data_dnmt3_mean.loc[data_dnmt3_mean['Activity'] == 'Active']
print(dnmt3_actives_max.shape[0])
print(dnmt3_actives_mean.shape[0])

### Obtener el desbalanceo de clases cuantitativamente
n_actives_max = data_max.loc[data_max['Activity'] == 'Active'].shape[0]
n_actives_mean = data_mean.loc[data_mean['Activity'] == 'Active'].shape[0]
n_inactives_max = data_max.shape[0] - n_actives_max
n_inactives_mean = data_mean.shape[0] - n_actives_mean

n_imbalance_max = n_inactives_max - n_actives_max
n_imbalance_mean = n_inactives_mean - n_actives_mean

### Balancear los sets originales
if (n_imbalance_max >= dnmt3_actives_max.shape[0]): data_dnmt1_3_max = data_max.append(dnmt3_actives_max)
else: data_dnmt1_3_max = data_max.append(dnmt3_actives_max.sample(n = n_imbalance_max))
if (n_imbalance_mean >= dnmt3_actives_mean.shape[0]): data_dnmt1_3_mean = data_mean.append(dnmt3_actives_mean)
else: data_dnmt1_3_mean = data_mean.append(dnmt3_actives_mean.sample(n = n_imbalance_mean))

## Oleacin
dataO_max = data_max.append(oleacin)
dataO_mean = data_mean.append(oleacin)

print("max: [", data_max.loc[data_max['Activity'] == 'Active'].shape[0], ",", data_max.loc[data_max['Activity'] == 'Inactive'].shape[0], "]")
print("mean: [", data_mean.loc[data_mean['Activity'] == 'Active'].shape[0], ",", data_mean.loc[data_mean['Activity'] == 'Inactive'].shape[0], "]")

print("3_max: [", data_dnmt1_3_max.loc[data_dnmt1_3_max['Activity'] == 'Active'].shape[0], ",", data_dnmt1_3_max.loc[data_dnmt1_3_max['Activity'] == 'Inactive'].shape[0], "]")
print("3_mean: [", data_dnmt1_3_mean.loc[data_dnmt1_3_mean['Activity'] == 'Active'].shape[0], ",", data_dnmt1_3_mean.loc[data_dnmt1_3_mean['Activity'] == 'Inactive'].shape[0], "]")

print("O_max: [", dataO_max.loc[dataO_max['Activity'] == 'Active'].shape[0], ",", dataO_max.loc[dataO_max['Activity'] == 'Inactive'].shape[0], "]")
print("O_mean: [", dataO_mean.loc[dataO_mean['Activity'] == 'Active'].shape[0], ",", dataO_mean.loc[dataO_mean['Activity'] == 'Inactive'].shape[0], "]")

# Preparación de los conjuntos de entrenamiento

## Dividir los conjuntos de datos en train y test
### DNMT1
train_max_X, test_max_X, train_max_Y, test_max_Y, max_X, max_Y = qm.splitData(data_max, 'train_max')
train_mean_X, test_mean_X, train_mean_Y, test_mean_Y, mean_X, mean_Y = qm.splitData(data_mean, 'train_mean')

### DNMT3
train3_max_X, test3_max_X, train3_max_Y, test3_max_Y, max3_X, max3_Y = qm.splitData(data_dnmt1_3_max, 'train3_max')
train3_mean_X, test3_mean_X, train3_mean_Y, test3_mean_Y, mean3_X, mean3_Y = qm.splitData(data_dnmt1_3_mean, 'train3_mean')

### Oleacin
trainO_max_X, testO_max_X, trainO_max_Y, testO_max_Y, maxO_X, maxO_Y = qm.splitData(dataO_max, 'trainO_max')
trainO_mean_X, testO_mean_X, trainO_mean_Y, testO_mean_Y, meanO_X, meanO_Y = qm.splitData(dataO_mean, 'trainO_mean')

print("test_max: [", len([i for i in test_max_Y if i == 1]), ",", len([i for i in test_max_Y if i == 0]), "]")
print("test_mean: [", len([i for i in test_mean_Y if i == 1]), ",", len([i for i in test_mean_Y if i == 0]), "]")

print("test_3_max: [", len([i for i in test3_max_Y if i == 1]), ",", len([i for i in test3_max_Y if i == 0]), "]")
print("test_3_mean: [", len([i for i in test3_mean_Y if i == 1]), ",", len([i for i in test3_mean_Y if i == 0]), "]")

print("test_O_max: [", len([i for i in testO_max_Y if i == 1]), ",", len([i for i in testO_max_Y if i == 0]), "]")
print("test_O_mean: [", len([i for i in testO_mean_Y if i == 1]), ",", len([i for i in testO_mean_Y if i == 0]), "]")

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








