# Cargar las librerías
import pandas as pd
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from tensorflow import keras
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

### Función que obtiene el listado de moléculas a partir de sus smiles (conservando su ID) ###
def getMols(smiles_df):
    return [(ids, Chem.MolFromSmiles(sml)) for ids, sml in smiles_df.itertuples(index=False)]

### Función que obtiene los Fingerprints de una lista de moléculas y los devuelve como DataFrame (conservando su ID) ###   
def getMorganFP(mols_df):
    # Obtener los Fingerprints + ID en una lista de Series
    m_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_FPs = [pd.concat([pd.Series(ids), pd.Series(np.array(m_generator.GetFingerprint(mol)))], ignore_index=True) for ids, mol in mols_df]
    
    # Generar los nombres de las columnas
    names = [f'FP_{n}' for n in range(2048)]
    names.insert(0, "Molecule ChEMBL ID")
    
    # Generar el DataFrame final con los datos en forma de columnas
    df_morgan_FPs = pd.DataFrame(data=morgan_FPs)
    df_morgan_FPs.set_axis(names, axis=1, inplace=True)
    return df_morgan_FPs

### Función que genera una red neuronal artificial con los parámetros establecidos ###
def buildANN(n):
    model = keras.Sequential([
        Dense(n, activation='relu', input_shape=(2048,)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Función que devuelve las métricas obtenidas a partir de la predicción de un modelo ###
def getQualityMetrics(y, pred):
    # Obtener la sensibilidad y la especificidad
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    sens = round(tp / (tp+fn),2)
    spec = round(tn / (tn+fp),2)
    
    # Obtener el valor del área bajo la curva ROC
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auc = round(metrics.auc(fpr, tpr),2)
    
    # Obtener las demás métricas
    fm = round(metrics.f1_score(y, pred),2)
    ba = round(metrics.balanced_accuracy_score(y, pred),2)
    
    # Devolver una fila con los datos
    return pd.Series([ba, sens, spec, tp, fp, fm, auc])

### Función que devuelve el número de True Positives ###
def tp_score(y, pred):
    _, _, _, tp = metrics.confusion_matrix(y, pred).ravel()
    return tp

### Función que devuelve el número de False Positives ###
def fp_score(y, pred):
    _, fp, _, _ = metrics.confusion_matrix(y, pred).ravel()
    return fp

### Función que entrena un modelo mediante tunning ###
def trainModel(model, params, kfold, trainX, trainY):
    # Establecer un scoring con métricas personalizadas
    scoring = {
        'balanced_accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),
        'sensitivity': metrics.make_scorer(metrics.recall_score),
        'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0),
        'true_positives' : metrics.make_scorer(tp_score),
        'false_positives' : metrics.make_scorer(fp_score),
        'F-measure' : metrics.make_scorer(metrics.f1_score),
        'AUC' : metrics.make_scorer(metrics.roc_auc_score)
    }
    
    # Realizar la búsqueda de los mejores hiperparámetros del modelo
    grid = GridSearchCV(estimator=model, param_grid=params, cv=kfold, scoring=scoring, refit="balanced_accuracy")
    grid.fit(trainX, trainY)
    
    # Realizar la predicción con el mejor modelo
    best_model = grid.best_estimator_
    
    # Obtener las métricas de calidad para el entrenamiento
    ind = grid.best_index_
    scores = pd.Series([round(grid.cv_results_['mean_test_balanced_accuracy'][ind],2), round(grid.cv_results_['mean_test_sensitivity'][ind],2),
                        round(grid.cv_results_['mean_test_specificity'][ind],2), round(grid.cv_results_['mean_test_true_positives'][ind],2),
                        round(grid.cv_results_['mean_test_false_positives'][ind],2), round(grid.cv_results_['mean_test_F-measure'][ind],2), 
                        round(grid.cv_results_['mean_test_AUC'][ind],2)])
    
    # Crear un dataframe auxiliar con las métricas como columnas
    df_metrs = pd.DataFrame()
    df_metrs = df_metrs.append(scores, ignore_index=True)
    
    # Devolver la combinación de parámetros ganadora y sus métricas para la fase de entrenamiento y de test
    return grid.best_params_, df_metrs, best_model
