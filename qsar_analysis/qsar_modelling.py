# Cargar las librer�as
import pandas as pd
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from tensorflow import keras
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

### Funci�n que obtiene el listado de mol�culas a partir de sus smiles (conservando su ID) ###
def getMols(smiles_df):
    return [(ids, Chem.MolFromSmiles(sml)) for ids, sml in smiles_df.itertuples(index=False)]

### Funci�n que obtiene los Fingerprints de una lista de mol�culas y los devuelve como DataFrame (conservando su ID) ###   
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

### Funci�n que genera una red neuronal artificial con los par�metros establecidos ###
def buildANN(n1, n2):
    model = keras.Sequential([
        Dense(n1, activation='relu', input_shape=(2048,)),
        Dense(n2, activation='relu'),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Funci�n que devuelve las m�tricas obtenidas a partir de la predicci�n de un modelo ###
def getQualityMetrics(y, pred):
    # Obtener la sensibilidad y la especificidad
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)
    
    # Obtener el valor del �rea bajo la curva ROC
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    # Obtener las dem�s m�tricas
    fm = metrics.f1_score(y, pred)
    ba = metrics.balanced_accuracy_score(y, pred)
    
    # Devolver una fila con los datos
    return pd.Series([ba, sens, spec, tp, fp, fm, auc])

def tp_score(y, pred):
    _, _, _, tp = metrics.confusion_matrix(y, pred).ravel()
    return tp

def fp_score(y, pred):
    _, fp, _, _ = metrics.confusion_matrix(y, pred).ravel()
    return fp

def trainTest(model, params, kfold, trainX, trainY, testX, testY):
    # Establecer un scoring con m�tricas personalizadas
    scoring = {
        'balanced_accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),
        'sensitivity': metrics.make_scorer(metrics.recall_score),
        'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0),
        'true_positives' : metrics.make_scorer(tp_score),
        'false_positives' : metrics.make_scorer(fp_score),
        'F-measure' : metrics.make_scorer(metrics.f1_score),
        'AUC' : metrics.make_scorer(metrics.roc_auc_score)
    }
    
    # Realizar la b�squeda de los mejores hiperpar�metros del modelo
    grid = GridSearchCV(estimator=model, param_grid=params, cv=kfold, scoring=scoring, refit="balanced_accuracy")
    grid.fit(trainX, trainY)
    
    # Realizar la predicci�n con el mejor modelo
    best_model = grid.best_estimator_
    pred = best_model.predict(testX)
    
    # Obtener las m�tricas de calidad para esta predicci�n
    qm = getQualityMetrics(testY, pred)
    
    # Obtener las m�tricas de calidad para el entrenamiento
    ind = grid.best_index_
    scores = pd.Series([grid.cv_results_['mean_test_balanced_accuracy'][ind], grid.cv_results_['mean_test_sensitivity'][ind],
                        grid.cv_results_['mean_test_specificity'][ind], grid.cv_results_['mean_test_true_positives'][ind],
                        grid.cv_results_['mean_test_false_positives'][ind], grid.cv_results_['mean_test_F-measure'][ind], 
                        grid.cv_results_['mean_test_AUC'][ind]])
    
    # Crear un dataframe auxiliar con las m�tricas como columnas
    df_metrs = pd.DataFrame()
    df_metrs = df_metrs.append(scores, ignore_index=True)
    df_metrs = df_metrs.append(qm, ignore_index=True)
    
    # Devolver la combinaci�n de par�metros ganadora y sus m�tricas para la fase de entrenamiento y de test
    return grid.best_params_, df_metrs
