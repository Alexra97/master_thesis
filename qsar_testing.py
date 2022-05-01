# Cargar las librerias
import sys
import pickle
from tensorflow import keras
import pandas as pd
import numpy as np
from qsar_pckg import qsar_modelling as qm
from qsar_pckg import chembl_screening as cs

# Cargar los datos del script main

## Importar los modelos
rf_model_max = pickle.load(open('rf_model_max.sav', 'rb'))
rf_model_mean = pickle.load(open('rf_model_mean.sav', 'rb'))
svm_model_max = pickle.load(open('svm_model_max.sav', 'rb'))
svm_model_mean = pickle.load(open('svm_model_mean.sav', 'rb'))
nb_model_max = pickle.load(open('nb_model_max.sav', 'rb'))
nb_model_mean = pickle.load(open('nb_model_mean.sav', 'rb'))
xgb_model_max = pickle.load(open('xgb_model_max.sav', 'rb'))
xgb_model_mean = pickle.load(open('xgb_model_mean.sav', 'rb'))
ann_model_max = keras.models.load_model('ann_model_max.h5')
ann_model_mean = keras.models.load_model('ann_model_mean.h5')

## Importar el conjunto de test
test_max_X = pd.read_csv('test_max_X.csv')
test_max_Y = pd.read_csv('test_max_Y.csv')
test_mean_X = pd.read_csv('test_mean_X.csv')
test_mean_Y = pd.read_csv('test_mean_Y.csv')

# Evaluar los modelos

## Crear listas de los modelos para iterarlos
models_max = [rf_model_max, svm_model_max, nb_model_max, xgb_model_max]
models_mean = [rf_model_mean, svm_model_mean, nb_model_mean, xgb_model_mean]

## Definir un dataframe que almacenar los resultados
results_test = pd.DataFrame()

## Evaluar cada modelo en su conjunto de test
for i in range(len(models_max)):
    ## Predecir sobre los conjuntos de test con el modelo adecuado
    pred_max = models_max[i].predict(test_max_X)
    pred_mean = models_mean[i].predict(test_mean_X)

    ## Obtener las métricas de calidad de las predicciones
    qm_max = qm.getQualityMetrics(test_max_Y, pred_max)
    qm_mean = qm.getQualityMetrics(test_mean_Y, pred_mean)

    ## Añadir los resultados al dataframe auxiliar
    results_test = results_test.append(qm_max, ignore_index=True)
    results_test = results_test.append(qm_mean, ignore_index=True)
    
## Evaluar de forma diferente ANN (produce valores contínuos)
### Predecir sobre los conjuntos de test
pred_max = np.around(ann_model_max.predict(test_max_X)).astype(int)
pred_mean = np.around(ann_model_mean.predict(test_mean_X)).astype(int)

### Obtener las métricas de calidad de las predicciones
qm_max = qm.getQualityMetrics(test_max_Y, pred_max)
qm_mean = qm.getQualityMetrics(test_mean_Y, pred_mean)

### Añadir los resultados al dataframe auxiliar
results_test = results_test.append(qm_max, ignore_index=True)
results_test = results_test.append(qm_mean, ignore_index=True)

# Almacenar los resultados

## Añadir los nombres de las columnas del df
results_test.columns = ['Balanced Accuracy', 'Sensitivity', 'Specificity', 'True Positives', 'False Positives', 'F-measure', 'AUC']

## Añadir columnas descriptivas
results_test.insert(0, "Algorithm", [ a for a in ['Random Forest', 'Support Vector Machines', 'Naive Bayes', 'XGradientBoostTree', 'Artificial Neural Network']
                                      for i in range(2)], True)

results_test.insert(1, "Version", [ s for i in range(5) for s in ['Max', 'Mean']], True)

## Exportar el dataframe a un .CSV
results_test.to_csv('results_test.csv', index=False)

# Cargar y transformar los datos del estudio por Docking
dock_data = pd.read_excel(sys.argv[1])
data_dnmt1_dock = cs.getSmiles(dock_data, 1)
data_dnmt1_dock.to_csv('data_dnmt1_dock.csv', index=False)


