# Cargar las librerías
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from keras.wrappers.scikit_learn import KerasClassifier
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
max_X = data_max.drop(["Molecule ChEMBL ID", "Activity"], axis=1)
mean_X = data_mean.drop(["Molecule ChEMBL ID", "Activity"], axis=1)
max_Y = data_max["Activity"].cat.codes
mean_Y = data_mean["Activity"].cat.codes

train_max_X, test_max_X, train_max_Y, test_max_Y = train_test_split(max_X, max_Y, test_size=0.2, stratify=max_Y)
train_mean_X, test_mean_X, train_mean_Y, test_mean_Y = train_test_split(mean_X, mean_Y, test_size=0.2, stratify=mean_Y)

## Definir el método de crosvalidación, en este caso 5 pliegues estratificados
kf = StratifiedKFold(5)

## Obtener las métricas de calidad para cada modelo (entrenamiento y test)

### Random Forest
#### Max
params_rf = {'bootstrap': [True, False],
             'max_depth': [5, 7, 10, None],
             'n_estimators': [50, 100, 200, 300, 500]}

rf_best_params_max, rf_metrs_max = qm.trainTest(ensemble.RandomForestClassifier(), params_rf, kf, train_max_X, train_max_Y, test_max_X, test_max_Y)
print("Parámetros ganadores para RF_max: ",rf_best_params_max)

#### Mean
rf_best_params_mean, rf_metrs_mean = qm.trainTest(ensemble.RandomForestClassifier(), params_rf, kf, train_mean_X, train_mean_Y, test_mean_X, test_mean_Y)
print("Parámetros ganadores para RF_mean: ",rf_best_params_mean)


### Support Vector Machines
#### Max
params_svm = {'C': [0.1, 0.5, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

svm_best_params_max, svm_metrs_max = qm.trainTest(SVC(), params_svm, kf, train_max_X, train_max_Y, test_max_X, test_max_Y)
print("Parámetros ganadores para SVM_max: ",svm_best_params_max)

#### Mean
svm_best_params_mean, svm_metrs_mean = qm.trainTest(SVC(), params_svm, kf, train_mean_X, train_mean_Y, test_mean_X, test_mean_Y)
print("Parámetros ganadores para SVM_mean: ",svm_best_params_mean)


### Naive Bayes
#### Max
params_nb = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}

nb_best_params_max, nb_metrs_max = qm.trainTest(BernoulliNB(), params_nb, kf, train_max_X, train_max_Y, test_max_X, test_max_Y)
print("Parámetros ganadores para NB_max: ",nb_best_params_max)

#### Mean
nb_best_params_mean, nb_metrs_mean = qm.trainTest(BernoulliNB(), params_nb, kf, train_mean_X, train_mean_Y, test_mean_X, test_mean_Y)
print("Parámetros ganadores para NB_mean: ",nb_best_params_mean)


### XGradientBoostTree
#### Max
params_xgb = {'min_child_weight': [1, 5, 10],
              'gamma': [0.5, 1, 1.5, 2, 5],
              'subsample': [0.6, 0.8, 1.0],
              'max_depth': [3, 4, 5]}

xgb_best_params_max, xgb_metrs_max = qm.trainTest(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), params_xgb, 
                                                  kf, train_max_X, train_max_Y, test_max_X, test_max_Y)
print("Parámetros ganadores para XGB_max: ",xgb_best_params_max)

#### Mean
xgb_best_params_mean, xgb_metrs_mean = qm.trainTest(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), params_xgb, 
                                                    kf, train_mean_X, train_mean_Y, test_mean_X, test_mean_Y)
print("Parámetros ganadores para XGB_mean: ",xgb_best_params_mean)


### Artificial Neural Network
#### Max
params_ann = {'batch_size': [10, 20, 50, 80],
              'epochs': [10, 50, 100],
              'n1': [10, 50, 100, 200],
              'n2': [10, 50, 100, 200]}

ann_best_params_max, ann_metrs_max = qm.trainTest(KerasClassifier(build_fn=qm.buildANN, verbose=0), params_ann, kf, train_max_X, train_max_Y, test_max_X, test_max_Y)
print("Parámetros ganadores para ANN_max: ",ann_best_params_max)

#### Mean
ann_best_params_mean, ann_metrs_mean = qm.trainTest(KerasClassifier(build_fn=qm.buildANN, verbose=0), params_ann, kf, train_mean_X, train_mean_Y, test_mean_X, test_mean_Y)
print("Parámetros ganadores para ANN_mean: ",ann_best_params_mean)











